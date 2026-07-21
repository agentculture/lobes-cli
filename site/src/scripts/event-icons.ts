/**
 * Inline SVG icon shapes for the event-stream log (issue #151 t12).
 *
 * Every `IconId` (see realtime-events.ts) gets a genuinely distinct
 * silhouette here — never a colour-only variant of another icon. `fill`/
 * `stroke` are `currentColor` throughout, so a row's CSS colour tints the
 * icon automatically (colour is reinforcement layered on top of a shape
 * that already reads on its own — the a11y bar this component holds
 * itself to: colour alone is never the only signal).
 *
 * Built with plain DOM calls (`createElementNS`), not innerHTML — this
 * module has no build step of its own and runs the same way in jsdom
 * (tests) and a real browser.
 */

import type { IconId } from "./realtime-events";

const SVG_NS = "http://www.w3.org/2000/svg";

function el<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Record<string, string>
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, value);
  }
  return node;
}

type ShapeBuilder = (svg: SVGSVGElement) => void;

const STROKE = { stroke: "currentColor", "stroke-width": "1.6", fill: "none" };
const STROKE_ROUND = { ...STROKE, "stroke-linecap": "round", "stroke-linejoin": "round" };

const SHAPES: Record<IconId, ShapeBuilder> = {
  "session-open": (svg) => {
    svg.append(el("circle", { cx: "8", cy: "8", r: "5", fill: "currentColor" }));
  },
  "session-close": (svg) => {
    svg.append(el("circle", { cx: "8", cy: "8", r: "5", ...STROKE }));
  },
  "boundary-start": (svg) => {
    svg.append(el("path", { d: "M3 11 L8 4 L13 11", ...STROKE_ROUND }));
  },
  "boundary-stop": (svg) => {
    svg.append(el("path", { d: "M3 5 L8 12 L13 5", ...STROKE_ROUND }));
  },
  transcript: (svg) => {
    svg.append(
      el("line", { x1: "3", y1: "5", x2: "13", y2: "5", ...STROKE_ROUND }),
      el("line", { x1: "3", y1: "8", x2: "13", y2: "8", ...STROKE_ROUND }),
      el("line", { x1: "3", y1: "11", x2: "9", y2: "11", ...STROKE_ROUND })
    );
  },
  "response-start": (svg) => {
    svg.append(el("path", { d: "M3 8 L11 8 M7 4 L11 8 L7 12", ...STROKE_ROUND }));
  },
  "response-text": (svg) => {
    svg.append(el("path", { d: "M3 8.5 L6.5 12 L13 4.5", ...STROKE_ROUND }));
  },
  "response-audio": (svg) => {
    svg.append(
      el("line", { x1: "3.5", y1: "10", x2: "3.5", y2: "6", ...STROKE_ROUND }),
      el("line", { x1: "7", y1: "12", x2: "7", y2: "4", ...STROKE_ROUND }),
      el("line", { x1: "10.5", y1: "10.5", x2: "10.5", y2: "5.5", ...STROKE_ROUND }),
      el("line", { x1: "14", y1: "9", x2: "14", y2: "7", ...STROKE_ROUND })
    );
  },
  "response-done": (svg) => {
    svg.append(
      el("circle", { cx: "8", cy: "8", r: "5.4", ...STROKE }),
      el("path", { d: "M5.5 8.2 L7.2 10 L10.6 6", ...STROKE_ROUND })
    );
  },
  "response-interrupted": (svg) => {
    svg.append(el("path", { d: "M9 3 L4 9 L7.5 9 L6 13 L12 6.5 L8.5 6.5 Z", fill: "currentColor" }));
  },
  "error-config": (svg) => {
    svg.append(
      el("path", { d: "M8 3 L14 13 L2 13 Z", ...STROKE_ROUND }),
      el("line", { x1: "8", y1: "6.5", x2: "8", y2: "9.5", ...STROKE_ROUND }),
      el("circle", { cx: "8", cy: "11.2", r: "0.6", fill: "currentColor" })
    );
  },
  "error-vad": (svg) => {
    svg.append(
      el("circle", { cx: "8", cy: "8", r: "5", ...STROKE }),
      el("line", { x1: "4", y1: "4", x2: "12", y2: "12", ...STROKE_ROUND })
    );
  },
  // A malformed-frame glyph: brackets standing for "one wire event", with a
  // jagged break inside them rather than the slash the other error icons
  // share — this is the one failure that never reached a backend at all, so
  // it earns a visibly different silhouette, not just a different colour.
  "error-wire": (svg) => {
    svg.append(
      el("path", { d: "M5.4 3 L3 3 L3 13 L5.4 13", ...STROKE_ROUND }),
      el("path", { d: "M10.6 3 L13 3 L13 13 L10.6 13", ...STROKE_ROUND }),
      el("path", { d: "M9 5 L6.6 8.6 L8 8.6 L7 11 L10 7.4 L8.4 7.4 Z", fill: "currentColor" })
    );
  },
  "error-stt": (svg) => {
    svg.append(
      el("rect", { x: "3.2", y: "3.2", width: "9.6", height: "9.6", rx: "1.4", ...STROKE }),
      el("line", { x1: "4", y1: "4", x2: "12", y2: "12", ...STROKE_ROUND })
    );
  },
  "error-generate": (svg) => {
    svg.append(
      el("path", { d: "M8 2.5 L13.5 8 L8 13.5 L2.5 8 Z", ...STROKE }),
      el("line", { x1: "4.4", y1: "4.4", x2: "11.6", y2: "11.6", ...STROKE_ROUND })
    );
  },
  "error-tts": (svg) => {
    svg.append(
      el("path", {
        d: "M8 2.6 L13 5.3 V10.7 L8 13.4 L3 10.7 V5.3 Z",
        ...STROKE,
      }),
      el("line", { x1: "4.4", y1: "4.4", x2: "11.6", y2: "11.6", ...STROKE_ROUND })
    );
  },
  "error-timeout": (svg) => {
    svg.append(
      el("circle", { cx: "8", cy: "8", r: "5.2", ...STROKE }),
      el("path", { d: "M8 5.2 L8 8 L10.2 9.6", ...STROKE_ROUND })
    );
  },
  "conn-connecting": (svg) => {
    const c = el("circle", { cx: "8", cy: "8", r: "5", ...STROKE });
    c.setAttribute("stroke-dasharray", "2.4 2.2");
    svg.append(c);
  },
  "conn-connected": (svg) => {
    svg.append(
      el("circle", { cx: "8", cy: "8", r: "3.4", fill: "currentColor" }),
      el("circle", { cx: "8", cy: "8", r: "6", ...STROKE })
    );
  },
  "conn-disconnected": (svg) => {
    svg.append(
      el("path", { d: "M2.5 8 A5.5 5.5 0 0 1 7 2.7", ...STROKE_ROUND }),
      el("path", { d: "M9 13.3 A5.5 5.5 0 0 0 13.5 8", ...STROKE_ROUND })
    );
  },
  "conn-error": (svg) => {
    svg.append(
      el("circle", { cx: "8", cy: "8", r: "5", ...STROKE }),
      el("line", { x1: "5.8", y1: "5.8", x2: "10.2", y2: "10.2", ...STROKE_ROUND }),
      el("line", { x1: "10.2", y1: "5.8", x2: "5.8", y2: "10.2", ...STROKE_ROUND })
    );
  },
  unknown: (svg) => {
    svg.append(
      el("circle", { cx: "8", cy: "8", r: "5.2", ...STROKE }),
      el("text", {
        x: "8",
        y: "10.6",
        "text-anchor": "middle",
        "font-size": "6.5",
        fill: "currentColor",
        stroke: "none",
      })
    );
    svg.querySelector("text")!.textContent = "?";
  },

  // A mic capsule on a stand. Muted adds a slash across it — the one place a
  // pair of icons is deliberately near-identical, because muted/unmuted are
  // two states of ONE thing and reading them as a pair is the point. The
  // slash (not colour) is what distinguishes them.
  "mic-unmuted": (svg) => {
    svg.append(
      el("rect", { x: "6", y: "2.5", width: "4", height: "7", rx: "2", ...STROKE }),
      el("path", { d: "M4 8a4 4 0 0 0 8 0", ...STROKE_ROUND }),
      el("line", { x1: "8", y1: "12", x2: "8", y2: "14", ...STROKE_ROUND })
    );
  },
  "mic-muted": (svg) => {
    svg.append(
      el("rect", { x: "6", y: "2.5", width: "4", height: "7", rx: "2", ...STROKE }),
      el("path", { d: "M4 8a4 4 0 0 0 8 0", ...STROKE_ROUND }),
      el("line", { x1: "8", y1: "12", x2: "8", y2: "14", ...STROKE_ROUND }),
      el("line", { x1: "3", y1: "13", x2: "13", y2: "3", ...STROKE_ROUND })
    );
  },
};

/** Build one icon's SVG element. Never throws — an unmapped id falls back to `unknown`. */
export function buildIcon(iconId: IconId): SVGSVGElement {
  const svg = el("svg", {
    viewBox: "0 0 16 16",
    width: "16",
    height: "16",
    class: "es-icon",
    "aria-hidden": "true",
    focusable: "false",
  });
  const shape = SHAPES[iconId] ?? SHAPES.unknown;
  shape(svg);
  return svg;
}
