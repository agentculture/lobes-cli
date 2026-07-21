/**
 * The live /v1/realtime event-stream log — issue #151 t12.
 *
 * This module owns the rendering model: turning one raw wire event (or a
 * client-side connection-state change) into a DOM row, with every event
 * type and every named error code visually distinct, VAD boundary timing
 * made visible, and a disconnect kept structurally separate from a named
 * server error. It has no opinion on WHERE events come from — a caller
 * (a WebSocket `onmessage` handler, a fixture replay, a future proxy
 * island) hands it parsed JSON one message at a time via
 * `EventStreamController.pushEvent`.
 *
 * Two ways to feed this component (see the component doc comment in
 * EventStream.astro and this repo's site README for the full mounting
 * contract):
 *
 *   1. Direct import: `mountEventStream(el)` returns a controller with
 *      `pushEvent`/`setConnectionState`/`clear` — call these straight from
 *      whatever owns the WebSocket.
 *   2. Decoupled `window` CustomEvents (recommended for cross-island
 *      wiring, since it needs no shared import): dispatch
 *      `new CustomEvent("lobes:realtime-event", { detail: parsedJson })`
 *      and `new CustomEvent("lobes:connection-state", { detail: { state,
 *      detail? } })` on `window`. `EventStream.astro`'s own mount script
 *      listens for both and forwards them to its controller.
 *
 * Silence is not a signal this module renders. There is no timer, no
 * idle-tick, and no "nothing has happened" row anywhere in this file —
 * the whole point of the vad_unavailable/silence distinction (see the
 * module-level test suite) is structural: silence is the absence of a
 * `pushEvent` call, full stop.
 */

import { buildIcon } from "./event-icons";
import {
  CONNECTION_KIND_META,
  ERROR_KIND_META,
  EVENT_KIND_META,
  isKnownErrorCode,
  isKnownEventType,
  isClientEventType,
  CLIENT_EVENT_KIND_META,
  type ColourFamily,
  type ConnectionState,
  type IconId,
  type RawEvent,
} from "./realtime-events";

/** Oldest rows are pruned past this cap so a long dev session stays light. */
export const MAX_LOG_ROWS = 500;

export interface LogEntry {
  /** A stable per-row id — the event's own `event_id` when present. */
  id: string;
  /** The raw `type` string as received (kept even when not in EVENT_TYPES). */
  eventType: string;
  family: ColourFamily | "connection" | "unknown";
  icon: IconId;
  label: string;
  /** Short stage badge for error rows (e.g. "STT"), undefined otherwise. */
  badge?: string;
  timeText: string;
  detailText: string;
  errorCode?: string;
  /** Number of coalesced response.audio.delta chunks folded into this row. */
  deltaChunks?: number;
  raw: RawEvent | { connectionState: ConnectionState; detail?: string };
}

export interface EventStreamController {
  readonly root: HTMLElement;
  /** Feed one parsed server event. Returns the entry rendered, or `null` for a no-op. */
  pushEvent(raw: unknown): LogEntry | null;
  /** Update the connection-state banner. Never touches the event log's error rows. */
  setConnectionState(state: ConnectionState, detail?: string): void;
  /** Empty the log (rows + bookkeeping). Leaves the connection banner untouched. */
  clear(): void;
  /** A defensive copy of the currently-rendered entries, newest last. */
  readonly entries: readonly LogEntry[];
  /** The connection state currently shown on the banner. */
  readonly connectionState: ConnectionState;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * Timing text for a boundary event (`speech_started`/`speech_stopped`).
 *
 * Reads `at_ms` when the server provides it (quantised audio-stream time —
 * see realtime-events.ts's module doc) and labels it as such. When absent,
 * falls back to elapsed time since `session.created` computed from
 * `timestamp_ms`, and is explicit that this fallback is NOT audio-stream
 * time — the honesty condition this task exists to satisfy. Exported
 * standalone so this logic has direct unit-test coverage independent of
 * the DOM.
 */
export function formatBoundaryTiming(
  raw: RawEvent,
  sessionStartMs: number | null
): { text: string; isAudioStreamTime: boolean } {
  const atMs = raw.at_ms;
  if (typeof atMs === "number" && Number.isFinite(atMs)) {
    return { text: `audio t+${atMs}ms (32ms-quantised audio-stream time)`, isAudioStreamTime: true };
  }
  const ts = raw.timestamp_ms;
  if (typeof ts === "number" && Number.isFinite(ts) && sessionStartMs !== null) {
    const elapsed = Math.max(0, ts - sessionStartMs);
    return {
      text: `elapsed t+${elapsed}ms (wall-clock elapsed — server did not send at_ms)`,
      isAudioStreamTime: false,
    };
  }
  return { text: "(no timing available)", isAudioStreamTime: false };
}

/** Plain elapsed-time text for any non-boundary event. */
function formatElapsed(raw: RawEvent, sessionStartMs: number | null): string {
  const ts = raw.timestamp_ms;
  if (typeof ts === "number" && Number.isFinite(ts)) {
    if (sessionStartMs !== null) {
      return `t+${Math.max(0, ts - sessionStartMs)}ms`;
    }
    return `ts ${ts}`;
  }
  return "t+?ms";
}

function truncate(text: string, max = 160): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function summarizeConfig(config: unknown): string {
  if (!isRecord(config)) return "(no config)";
  const rate = config.input_sample_rate ?? "?";
  const aec = config.aec_mode ?? "?";
  const turnDetection = config.turn_detection ?? "?";
  return `${rate}Hz, turn_detection=${turnDetection}, aec_mode=${aec}`;
}

interface ClassifyContext {
  sessionStartMs: number | null;
}

interface Classified {
  family: ColourFamily | "unknown";
  icon: IconId;
  label: string;
  badge?: string;
  timeText: string;
  detailText: string;
  errorCode?: string;
}

function classifyEvent(raw: RawEvent, ctx: ClassifyContext): Classified {
  const type = raw.type;

  if (type === "error") {
    const code = typeof raw.code === "string" ? raw.code : "";
    const message = typeof raw.message === "string" ? raw.message : "(no message)";
    if (isKnownErrorCode(code)) {
      const meta = ERROR_KIND_META[code];
      return {
        family: "error",
        icon: meta.icon,
        label: meta.label,
        badge: meta.badge,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `${truncate(message)} — ${meta.hint}`,
        errorCode: code,
      };
    }
    return {
      family: "error",
      icon: "unknown",
      label: code ? `error (unrecognized code: ${code})` : "error (no code)",
      badge: "ERR",
      timeText: formatElapsed(raw, ctx.sessionStartMs),
      detailText: `${truncate(message)} — this UI's vocabulary does not know this code yet`,
      errorCode: code || undefined,
    };
  }

  // Client-origin rows (the mic island's mute/unmute, deviation d1). They are
  // NOT wire events and never claim to be: the row says "(you)" and the detail
  // names the origin, so a muted stretch of log can never be misread as the
  // server having gone quiet. This is the distinction the log exists to make —
  // muted, silence, and disconnected are three different nothings.
  if (isClientEventType(type)) {
    const meta = CLIENT_EVENT_KIND_META[type];
    return {
      family: meta.family,
      icon: meta.icon,
      label: meta.label,
      timeText: formatElapsed(raw, ctx.sessionStartMs),
      detailText:
        type === "client.mic_muted"
          ? "you muted the mic — the session is still open and still listening for the reply"
          : "you unmuted the mic",
    };
  }

  if (!isKnownEventType(type)) {
    return {
      family: "unknown",
      icon: "unknown",
      label: `unrecognized event: ${type}`,
      timeText: formatElapsed(raw, ctx.sessionStartMs),
      detailText: "this UI's vocabulary does not know this event type yet",
    };
  }

  const meta = EVENT_KIND_META[type];

  switch (type) {
    case "session.created":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `session_id=${raw.session_id ?? "?"} — ${summarizeConfig(raw.config)}`,
      };
    case "session.closed":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `reason: ${typeof raw.reason === "string" ? raw.reason : "?"}`,
      };
    case "input_audio_buffer.speech_started": {
      const timing = formatBoundaryTiming(raw, ctx.sessionStartMs);
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: timing.text,
        detailText: `item_id=${raw.item_id ?? "?"}`,
      };
    }
    case "input_audio_buffer.speech_stopped": {
      const timing = formatBoundaryTiming(raw, ctx.sessionStartMs);
      const reason = typeof raw.reason === "string" ? ` reason=${raw.reason}` : "";
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: timing.text,
        detailText: `item_id=${raw.item_id ?? "?"}${reason}`,
      };
    }
    case "conversation.item.input_audio_transcription.completed":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `"${truncate(typeof raw.text === "string" ? raw.text : "")}"`,
      };
    case "response.created":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `response_id=${raw.response_id ?? "?"} item_id=${raw.item_id ?? "—"}`,
      };
    case "response.text.done":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `"${truncate(typeof raw.text === "string" ? raw.text : "")}"`,
      };
    case "response.audio.delta":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: "1 chunk streamed",
      };
    case "response.done":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `response_id=${raw.response_id ?? "?"}`,
      };
    case "response.interrupted":
      return {
        family: meta.family,
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: `response_id=${raw.response_id ?? "?"} truncated=${raw.truncated ?? true}`,
      };
    default:
      // Exhaustiveness guard — every EventType above has its own case, so
      // reaching this means EVENT_TYPES grew a member this switch forgot.
      return {
        family: "unknown",
        icon: "unknown",
        label: `unhandled event: ${type}`,
        timeText: formatElapsed(raw, ctx.sessionStartMs),
        detailText: "renderer gap — file an issue",
      };
  }
}

export function mountEventStream(root: HTMLElement): EventStreamController {
  const logEl = root.querySelector<HTMLElement>("[data-event-log]");
  const emptyNoteEl = root.querySelector<HTMLElement>("[data-empty-note]");
  const connLabelEl = root.querySelector<HTMLElement>("[data-connection-label]");
  const connDotEl = root.querySelector<HTMLElement>("[data-connection-dot]");
  const connDetailEl = root.querySelector<HTMLElement>("[data-connection-detail]");
  if (!logEl) {
    throw new Error("mountEventStream: root is missing a [data-event-log] element");
  }

  let sessionStartMs: number | null = null;
  let connectionState: ConnectionState = "idle";
  let lastDelta: { responseId: string; row: HTMLElement; chunks: number } | null = null;
  const entries: LogEntry[] = [];

  function updateEmptyNote(): void {
    if (emptyNoteEl) emptyNoteEl.hidden = entries.length > 0;
  }

  function pruneIfNeeded(): void {
    while (entries.length > MAX_LOG_ROWS) {
      entries.shift();
      const first = logEl!.firstElementChild;
      if (first) first.remove();
    }
  }

  function buildRow(entry: LogEntry): HTMLElement {
    const row = document.createElement("li");
    row.className = `es-row kind-${entry.family}`;
    row.dataset.eventType = entry.eventType;
    row.dataset.icon = entry.icon;
    if (entry.errorCode) row.dataset.errorCode = entry.errorCode;
    if (entry.badge) row.dataset.badge = entry.badge;

    const iconWrap = document.createElement("span");
    iconWrap.className = "es-row-icon";
    iconWrap.append(buildIcon(entry.icon));
    row.append(iconWrap);

    const body = document.createElement("span");
    body.className = "es-row-body";

    const head = document.createElement("span");
    head.className = "es-row-head";
    if (entry.badge) {
      const badge = document.createElement("span");
      badge.className = "es-badge";
      badge.textContent = entry.badge;
      head.append(badge);
    }
    const label = document.createElement("span");
    label.className = "es-label";
    label.textContent = entry.label;
    head.append(label);
    const time = document.createElement("time");
    time.className = "es-time";
    time.textContent = entry.timeText;
    head.append(time);
    body.append(head);

    const detail = document.createElement("span");
    detail.className = "es-detail";
    detail.textContent = entry.detailText;
    body.append(detail);

    row.append(body);
    return row;
  }

  function appendEntry(entry: LogEntry): void {
    entries.push(entry);
    const row = buildRow(entry);
    logEl!.append(row);
    // Entrance motion only when the visitor allows it — gated in CSS, not
    // here; this class toggle is inert (no visual effect) under
    // prefers-reduced-motion: reduce, matching the [data-reveal] convention
    // in global.css.
    requestAnimationFrame(() => row.classList.add("es-row-in"));
    pruneIfNeeded();
    updateEmptyNote();
    if (entry.eventType === "response.audio.delta") {
      const responseId = typeof entry.raw === "object" && "response_id" in entry.raw
        ? String((entry.raw as RawEvent).response_id ?? "unknown")
        : "unknown";
      lastDelta = { responseId, row, chunks: 1 };
    } else {
      lastDelta = null;
    }
  }

  function updateDeltaRow(existing: { row: HTMLElement; chunks: number }): void {
    existing.chunks += 1;
    const detailEl = existing.row.querySelector<HTMLElement>(".es-detail");
    if (detailEl) detailEl.textContent = `${existing.chunks} chunks streamed`;
    const last = entries[entries.length - 1];
    if (last && last.eventType === "response.audio.delta") {
      last.deltaChunks = existing.chunks;
      last.detailText = `${existing.chunks} chunks streamed`;
    }
  }

  function pushEvent(raw: unknown): LogEntry | null {
    if (!isRecord(raw) || typeof raw.type !== "string") {
      const entry: LogEntry = {
        id: `malformed-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        eventType: "malformed",
        family: "unknown",
        icon: "unknown",
        label: "malformed event payload",
        timeText: formatElapsed({ type: "malformed" }, sessionStartMs),
        detailText: "received a message that was not a {type: string, ...} object",
        raw: isRecord(raw) ? (raw as RawEvent) : { type: "malformed" },
      };
      appendEntry(entry);
      return entry;
    }

    const event = raw as RawEvent;

    if (event.type === "session.created" && typeof event.timestamp_ms === "number") {
      sessionStartMs = event.timestamp_ms;
    }

    // response.audio.delta for the same in-flight response coalesces into
    // the existing row instead of flooding the log with one row per
    // ~100ms chunk (see DEFAULT_DELTA_CHUNK_BYTES in lobes/realtime/_wire.py).
    if (event.type === "response.audio.delta") {
      const responseId = typeof event.response_id === "string" ? event.response_id : "unknown";
      if (lastDelta && lastDelta.responseId === responseId && lastDelta.row.isConnected) {
        updateDeltaRow(lastDelta);
        return entries[entries.length - 1] ?? null;
      }
    }

    const classified = classifyEvent(event, { sessionStartMs });
    const entry: LogEntry = {
      id: typeof event.event_id === "string" ? event.event_id : `evt-${entries.length}`,
      eventType: event.type,
      family: classified.family,
      icon: classified.icon,
      label: classified.label,
      badge: classified.badge,
      timeText: classified.timeText,
      detailText: classified.detailText,
      errorCode: classified.errorCode,
      raw: event,
    };
    appendEntry(entry);
    return entry;
  }

  function setConnectionState(state: ConnectionState, detail?: string): void {
    const previous = connectionState;
    connectionState = state;
    const meta = CONNECTION_KIND_META[state];
    if (connLabelEl) connLabelEl.textContent = meta.label;
    if (connDotEl) connDotEl.dataset.state = state;
    root.dataset.connectionState = state;
    if (connDetailEl) {
      if (detail) {
        connDetailEl.textContent = detail;
        connDetailEl.hidden = false;
      } else {
        connDetailEl.hidden = true;
        connDetailEl.textContent = "";
      }
    }

    // A connection-state change gets its own neutral timeline marker (own
    // family, own icon set — never `kind-error`) so an operator can see
    // WHEN a disconnect happened relative to other events without ever
    // confusing it for a named server error. The resting "idle" state
    // (before any connect attempt) is not itself a transition and gets no
    // marker.
    if (state !== "idle" && state !== previous) {
      const entry: LogEntry = {
        id: `conn-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        eventType: "connection.state",
        family: "connection",
        icon: meta.icon,
        label: meta.label,
        timeText: formatElapsed({ type: "connection.state", timestamp_ms: undefined }, sessionStartMs),
        detailText: detail ?? "(client-side transport state — not a server event)",
        raw: { connectionState: state, detail },
      };
      appendEntry(entry);
    }
    if (state === "disconnected" || state === "error") {
      lastDelta = null;
    }
  }

  function clear(): void {
    entries.length = 0;
    logEl!.replaceChildren();
    lastDelta = null;
    sessionStartMs = null;
    updateEmptyNote();
  }

  updateEmptyNote();

  return {
    root,
    pushEvent,
    setConnectionState,
    clear,
    get entries() {
      return entries.slice();
    },
    get connectionState() {
      return connectionState;
    },
  };
}
