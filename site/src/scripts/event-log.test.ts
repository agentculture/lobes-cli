import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { beforeEach, describe, expect, it } from "vitest";
import { mountEventStream, formatBoundaryTiming, MAX_LOG_ROWS } from "./event-log";
import {
  EVENT_TYPES,
  ERROR_CODES,
  CONNECTION_STATES,
  type ConnectionState,
} from "./realtime-events";
import {
  EVENT_FIXTURES,
  CONNECTION_FIXTURES,
  CONNECTION_ERROR_FIXTURE,
} from "./event-fixtures";

/** Builds the exact DOM shape EventStream.astro renders, so mountEventStream
 * can be exercised the same way it will be in the real page. */
function createRoot(): HTMLElement {
  const root = document.createElement("div");
  root.innerHTML = `
    <div class="es-head">
      <span class="es-dot" data-connection-dot></span>
      <span class="es-title">/v1/realtime — event stream</span>
      <span class="es-conn" data-connection-label>Not connected</span>
      <button type="button" data-clear-log>Clear</button>
    </div>
    <p class="es-conn-detail" data-connection-detail hidden></p>
    <ol class="es-log" data-event-log aria-live="polite" aria-label="realtime event log"></ol>
    <p class="es-empty" data-empty-note>No events yet.</p>
  `;
  document.body.append(root);
  return root;
}

describe("mountEventStream — fixture coverage (core deliverable)", () => {
  it("EVENT_FIXTURES covers every EventType declared in realtime-events.ts", () => {
    const covered = new Set(EVENT_FIXTURES.map((e) => e.type));
    for (const type of EVENT_TYPES) {
      expect(covered.has(type), `no fixture for EventType "${type}"`).toBe(true);
    }
  });

  it("EVENT_FIXTURES covers every ErrorCode declared in realtime-events.ts", () => {
    const covered = new Set(
      EVENT_FIXTURES.filter((e) => e.type === "error").map((e) => e.code as string)
    );
    for (const code of ERROR_CODES) {
      expect(covered.has(code), `no fixture for ErrorCode "${code}"`).toBe(true);
    }
  });

  it("CONNECTION_FIXTURES + CONNECTION_ERROR_FIXTURE cover every ConnectionState", () => {
    const covered = new Set<ConnectionState>([
      ...CONNECTION_FIXTURES.map((c) => c.state),
      ...CONNECTION_ERROR_FIXTURE.map((c) => c.state),
      "idle", // the resting default, never dispatched as a transition
    ]);
    for (const state of CONNECTION_STATES) {
      expect(covered.has(state), `no fixture for ConnectionState "${state}"`).toBe(true);
    }
  });

  it("a full fixture replay renders one row per EventType/ErrorCode, each rendered distinctly", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    for (const event of EVENT_FIXTURES) view.pushEvent(event);

    // Every EventType produced at least one row tagged with its wire type.
    const allRows = Array.from(root.querySelectorAll<HTMLElement>(".es-row"));
    for (const type of EVENT_TYPES) {
      const rows = allRows.filter((row) => row.dataset.eventType === type);
      expect(rows.length, `expected at least one row for "${type}"`).toBeGreaterThan(0);
    }

    // Every ErrorCode produced its own row, tagged and never sharing a
    // label/badge/icon with another code.
    const errorRows = Array.from(root.querySelectorAll('[data-event-type="error"]'));
    expect(errorRows).toHaveLength(ERROR_CODES.length);
    const seenBadges = new Set<string>();
    const seenIcons = new Set<string>();
    const seenLabels = new Set<string>();
    for (const code of ERROR_CODES) {
      const row = root.querySelector<HTMLElement>(`[data-error-code="${code}"]`);
      expect(row, `no row for error code "${code}"`).not.toBeNull();
      const badge = row!.dataset.badge!;
      const icon = row!.dataset.icon!;
      const label = row!.querySelector(".es-label")!.textContent!;
      expect(seenBadges.has(badge), `badge "${badge}" reused across error codes`).toBe(false);
      expect(seenIcons.has(icon), `icon "${icon}" reused across error codes`).toBe(false);
      expect(seenLabels.has(label), `label "${label}" reused across error codes`).toBe(false);
      seenBadges.add(badge);
      seenIcons.add(icon);
      seenLabels.add(label);
    }
  });
});

// ---------------------------------------------------------------------------
// Drift guard — issue #151 t19.
//
// realtime-events.ts hand-mirrors lobes/realtime/_session.py's ErrorCode
// enum because site/ has no Python import path into lobes/ (see that
// module's own doc comment). A hand-mirror only stays honest if something
// checks it: this reads _session.py as plain text off disk (vitest runs in
// Node, so the filesystem is available) and asserts ERROR_CODES matches the
// server enum member-for-member, IN ORDER — exactly what the #151 t12
// mirror comment already promised but nothing enforced. Before this task,
// the server's new INVALID_WIRE_EVENT landed with no site-side fixture and
// no failing test; this is what makes the next such landing fail loudly
// instead.
// ---------------------------------------------------------------------------

/**
 * Pull every member value out of a Python `class <name>(str, Enum):` block,
 * in source order. Deliberately narrow — not a Python parser — matching only
 * the `ALL_CAPS = "value"` assignment lines the enum body itself contains,
 * so it does not also pick up quoted enum-member names mentioned in prose
 * inside the class docstring (those are written as `` ``NAME`` `` /
 * `` ``value`` ``, never as a bare `NAME = "value"` assignment).
 */
function extractPythonStrEnumValues(source: string, className: string): string[] {
  const classRe = new RegExp(`class ${className}\\(str, Enum\\):\\n([\\s\\S]*?)\\n(?=@|class )`);
  const match = source.match(classRe);
  if (!match) {
    throw new Error(
      `could not find "class ${className}(str, Enum):" in the given source — has _session.py been restructured?`
    );
  }
  return [...match[1]!.matchAll(/^\s+[A-Z][A-Z0-9_]*\s*=\s*"([a-z_]+)"/gm)].map((m) => m[1]!);
}

describe("ERROR_CODES — drift guard against lobes/realtime/_session.py", () => {
  it("matches _session.py's ErrorCode enum member-for-member, in order", () => {
    const sessionSource = readFileSync(
      resolve(process.cwd(), "../lobes/realtime/_session.py"),
      "utf8"
    );
    const serverCodes = extractPythonStrEnumValues(sessionSource, "ErrorCode");

    // Sanity check on the parser itself: if this ever comes back empty, the
    // test below would pass vacuously against an equally-empty ERROR_CODES
    // rather than failing loudly on a broken extractor.
    expect(serverCodes.length).toBeGreaterThan(0);

    expect(ERROR_CODES).toEqual(serverCodes);
  });
});

describe("mountEventStream — silence vs. vad_unavailable vs. disconnect", () => {
  it("silence (no pushEvent calls) renders nothing", () => {
    const root = createRoot();
    mountEventStream(root);
    expect(root.querySelectorAll(".es-row")).toHaveLength(0);
    expect(root.querySelector<HTMLElement>("[data-empty-note]")!.hidden).toBe(false);
  });

  it("vad_unavailable renders a distinct row where silence rendered none", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    expect(root.querySelectorAll(".es-row")).toHaveLength(0);

    view.pushEvent({
      type: "error",
      code: "vad_unavailable",
      message: "Silero failed to load",
      timestamp_ms: 1_000_000,
    });

    const rows = root.querySelectorAll(".es-row");
    expect(rows).toHaveLength(1);
    expect((rows[0] as HTMLElement).dataset.errorCode).toBe("vad_unavailable");
    expect(root.querySelector<HTMLElement>("[data-empty-note]")!.hidden).toBe(true);
  });

  it("a disconnect renders its own banner state, never a log row styled as an error", () => {
    const root = createRoot();
    const view = mountEventStream(root);

    view.pushEvent({ type: "session.created", timestamp_ms: 1_000_000, config: {} });
    view.pushEvent({
      type: "error",
      code: "generate_failed",
      message: "gateway 404",
      timestamp_ms: 1_000_500,
    });
    view.setConnectionState("connecting");
    view.setConnectionState("connected");
    view.setConnectionState("disconnected", "server closed the connection");

    expect(view.connectionState).toBe("disconnected");
    expect(root.dataset.connectionState).toBe("disconnected");
    expect(root.querySelector("[data-connection-label]")!.textContent).toBe("Disconnected");

    // No connection-family row is ever tagged kind-error, and no error row
    // is ever tagged kind-connection — the two families are structurally
    // disjoint, not just differently coloured.
    expect(root.querySelectorAll(".kind-connection.kind-error")).toHaveLength(0);
    const connectionRows = root.querySelectorAll(".kind-connection");
    for (const row of Array.from(connectionRows)) {
      expect((row as HTMLElement).dataset.errorCode).toBeUndefined();
    }
    const errorRows = root.querySelectorAll(".kind-error");
    for (const row of Array.from(errorRows)) {
      expect(row.className).not.toContain("kind-connection");
    }
    // The disconnect did not remove or relabel the earlier named error row.
    expect(root.querySelectorAll('[data-error-code="generate_failed"]')).toHaveLength(1);
  });

  it("a transport error (connection state) is distinct from a named server error", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.setConnectionState("connecting");
    view.setConnectionState("error", "WebSocket error");

    const connectionErrorRows = root.querySelectorAll(".kind-connection");
    expect(connectionErrorRows.length).toBeGreaterThan(0);
    for (const row of Array.from(connectionErrorRows)) {
      expect((row as HTMLElement).dataset.eventType).not.toBe("error");
    }
    // No wire `error` event was ever pushed.
    expect(root.querySelectorAll('[data-event-type="error"]')).toHaveLength(0);
  });
});

describe("mountEventStream — stage-distinct error rendering", () => {
  it.each(ERROR_CODES)("renders %s with its own badge, icon, and label", (code) => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "error", code, message: "boom", timestamp_ms: 1_000_000 });
    const row = root.querySelector<HTMLElement>(`[data-error-code="${code}"]`)!;
    expect(row).not.toBeNull();
    expect(row.dataset.badge).toBeTruthy();
    expect(row.dataset.icon).toBeTruthy();
    expect(row.querySelector(".es-label")!.textContent).toContain(code);
  });

  it("an unrecognized error code still renders (never dropped), flagged as unknown", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    const entry = view.pushEvent({
      type: "error",
      code: "some_future_error_code",
      message: "a code this UI has never seen",
      timestamp_ms: 1_000_000,
    });
    expect(entry).not.toBeNull();
    const row = root.querySelector<HTMLElement>('[data-error-code="some_future_error_code"]');
    expect(row).not.toBeNull();
    expect(row!.querySelector(".es-detail")!.textContent).toContain("does not know this code yet");
  });
});

describe("mountEventStream — VAD boundary timing (at_ms observability)", () => {
  it("renders the server's at_ms, explicitly labelled as audio-stream time", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "session.created", timestamp_ms: 1_000_000, config: {} });
    view.pushEvent({
      type: "input_audio_buffer.speech_started",
      item_id: "item_1",
      at_ms: 288,
      timestamp_ms: 1_000_050,
    });
    const row = root.querySelector<HTMLElement>(
      '[data-event-type="input_audio_buffer.speech_started"]'
    )!;
    const timeText = row.querySelector(".es-time")!.textContent!;
    expect(timeText).toContain("288");
    expect(timeText.toLowerCase()).toContain("audio-stream");
  });

  it("falls back to elapsed wall-clock time when at_ms is absent, and says so honestly", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "session.created", timestamp_ms: 1_000_000, config: {} });
    view.pushEvent({
      type: "input_audio_buffer.speech_stopped",
      item_id: "item_1",
      timestamp_ms: 1_002_400,
      // no at_ms — mirrors the current, pre-#151-t6 server wire.
    });
    const row = root.querySelector<HTMLElement>(
      '[data-event-type="input_audio_buffer.speech_stopped"]'
    )!;
    const timeText = row.querySelector(".es-time")!.textContent!;
    expect(timeText).toContain("2400");
    expect(timeText.toLowerCase()).not.toContain("audio-stream");
    expect(timeText.toLowerCase()).toContain("wall-clock");
  });

  it("formatBoundaryTiming is a pure function covering both branches directly", () => {
    const withAtMs = formatBoundaryTiming(
      { type: "input_audio_buffer.speech_started", at_ms: 512 },
      1_000_000
    );
    expect(withAtMs.isAudioStreamTime).toBe(true);
    expect(withAtMs.text).toContain("512");

    const withoutAtMs = formatBoundaryTiming(
      { type: "input_audio_buffer.speech_started", timestamp_ms: 1_000_900 },
      1_000_000
    );
    expect(withoutAtMs.isAudioStreamTime).toBe(false);
    expect(withoutAtMs.text).toContain("900");
  });

  it("a max_turn force-commit and a silence-confirmed stop both render, distinguishably, by reason", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "session.created", timestamp_ms: 1_000_000, config: {} });
    view.pushEvent({
      type: "input_audio_buffer.speech_stopped",
      item_id: "item_1",
      at_ms: 2048,
      reason: "silence",
      timestamp_ms: 1_002_400,
    });
    view.pushEvent({
      type: "input_audio_buffer.speech_stopped",
      item_id: "item_2",
      at_ms: 39000,
      reason: "max_turn",
      timestamp_ms: 1_040_000,
    });
    const details = Array.from(
      root.querySelectorAll('[data-event-type="input_audio_buffer.speech_stopped"] .es-detail')
    ).map((el) => el.textContent);
    expect(details.some((d) => d?.includes("reason=silence"))).toBe(true);
    expect(details.some((d) => d?.includes("reason=max_turn"))).toBe(true);
  });
});

describe("mountEventStream — response.audio.delta coalescing", () => {
  it("coalesces successive deltas for the same response into one updating row", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "response.created", response_id: "resp_1", timestamp_ms: 1_000_000 });
    for (let i = 0; i < 5; i++) {
      view.pushEvent({ type: "response.audio.delta", response_id: "resp_1", delta: "AAAA" });
    }
    const rows = root.querySelectorAll('[data-event-type="response.audio.delta"]');
    expect(rows).toHaveLength(1);
    expect(rows[0].querySelector(".es-detail")!.textContent).toContain("5 chunks streamed");
  });

  it("starts a fresh row when a new response_id begins streaming", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "response.audio.delta", response_id: "resp_1", delta: "AAAA" });
    view.pushEvent({ type: "response.audio.delta", response_id: "resp_1", delta: "AAAA" });
    view.pushEvent({ type: "response.audio.delta", response_id: "resp_2", delta: "AAAA" });
    const rows = root.querySelectorAll('[data-event-type="response.audio.delta"]');
    expect(rows).toHaveLength(2);
    expect(rows[0].querySelector(".es-detail")!.textContent).toContain("2 chunks streamed");
    expect(rows[1].querySelector(".es-detail")!.textContent).toContain("1 chunk");
  });

  it("an intervening non-delta event ends coalescing for the next delta of the same response", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.pushEvent({ type: "response.audio.delta", response_id: "resp_1", delta: "AAAA" });
    view.pushEvent({ type: "response.interrupted", response_id: "resp_1", truncated: true });
    view.pushEvent({ type: "response.audio.delta", response_id: "resp_1", delta: "AAAA" });
    const rows = root.querySelectorAll('[data-event-type="response.audio.delta"]');
    expect(rows).toHaveLength(2);
  });
});

describe("mountEventStream — robustness", () => {
  it("never throws on a malformed payload and renders a distinct row for it", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    expect(() => view.pushEvent("not an object")).not.toThrow();
    expect(() => view.pushEvent(null)).not.toThrow();
    expect(() => view.pushEvent({ no_type_field: true })).not.toThrow();
    expect(root.querySelectorAll(".es-row")).toHaveLength(3);
  });

  it("clear() empties the log but leaves the connection banner untouched", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    view.setConnectionState("connected");
    view.pushEvent({ type: "session.created", timestamp_ms: 1_000_000, config: {} });
    view.pushEvent({ type: "error", code: "vad_unavailable", message: "x", timestamp_ms: 1 });
    expect(view.entries.length).toBeGreaterThan(0);

    view.clear();

    expect(view.entries).toHaveLength(0);
    expect(root.querySelectorAll(".es-row")).toHaveLength(0);
    expect(view.connectionState).toBe("connected");
    expect(root.querySelector("[data-connection-label]")!.textContent).toBe("Connected");
  });

  it("prunes the oldest rows once MAX_LOG_ROWS is exceeded", () => {
    const root = createRoot();
    const view = mountEventStream(root);
    for (let i = 0; i < MAX_LOG_ROWS + 25; i++) {
      view.pushEvent({ type: "session.closed", reason: `n${i}`, timestamp_ms: i });
    }
    expect(root.querySelectorAll(".es-row")).toHaveLength(MAX_LOG_ROWS);
    expect(view.entries).toHaveLength(MAX_LOG_ROWS);
    // The oldest entries were dropped, not the newest.
    const lastDetail = root.querySelector(".es-row:last-child .es-detail")!.textContent;
    expect(lastDetail).toContain(`n${MAX_LOG_ROWS + 24}`);
  });
});

describe("event-fixtures.ts — replay drives the same controller cleanly", () => {
  it("replays the full story without throwing, ending on session.closed", async () => {
    const root = createRoot();
    const view = mountEventStream(root);
    for (const transition of CONNECTION_FIXTURES) {
      view.setConnectionState(transition.state, transition.detail);
    }
    for (const event of EVENT_FIXTURES) {
      expect(() => view.pushEvent(event)).not.toThrow();
    }
    const last = view.entries[view.entries.length - 1];
    expect(last.eventType).toBe("session.closed");
  });
});

beforeEach(() => {
  document.body.innerHTML = "";
});

// ---------------------------------------------------------------------------
// Client-origin rows — deviation d1 (coordinator wiring).
//
// The mic island emits these when the OPERATOR mutes; the server never sends
// them. They exist so the log can tell apart three things that all look like
// nothing happening: muted (you did it), silence (nobody spoke — emits no
// event at all, by design), and disconnected (a banner, not a row).
// ---------------------------------------------------------------------------
describe("mountEventStream — client-origin mute rows", () => {
  it("renders a mute as its own kind, not the unrecognized-event fallback", () => {
    const controller = mountEventStream(createRoot());
    const entry = controller.pushEvent({
      type: "client.mic_muted",
      origin: "client",
      timestamp_ms: 1000,
    });

    expect(entry).not.toBeNull();
    expect(entry!.family).toBe("client");
    expect(entry!.label).not.toContain("unrecognized");
    expect(entry!.icon).toBe("mic-muted");
  });

  it("says who caused it, so a muted stretch is never read as a quiet server", () => {
    const controller = mountEventStream(createRoot());
    const muted = controller.pushEvent({ type: "client.mic_muted", timestamp_ms: 1000 })!;
    const unmuted = controller.pushEvent({ type: "client.mic_unmuted", timestamp_ms: 2000 })!;

    expect(muted.label).toContain("you");
    expect(muted.detailText).toContain("still open");
    expect(unmuted.label).toContain("you");
    expect(unmuted.icon).not.toBe(muted.icon);
  });

  it("keeps muted, silence and disconnected structurally distinct", () => {
    const root = createRoot();
    const controller = mountEventStream(root);

    // Silence: no pushEvent call at all. The log stays empty — there is no
    // "nothing happened" row anywhere in this module, and that is the point.
    expect(controller.entries).toHaveLength(0);

    controller.pushEvent({ type: "client.mic_muted", timestamp_ms: 1000 });
    expect(controller.entries).toHaveLength(1);

    // A disconnect gets its own row AND the banner — and its row belongs to a
    // different family than the mute, so the two nothings never blur together.
    controller.setConnectionState("disconnected", "socket closed");
    expect(controller.connectionState).toBe("disconnected");
    expect(controller.entries.map((entry) => entry.family)).toEqual(["client", "connection"]);
  });

  it("does not collide with the server vocabulary", () => {
    for (const clientType of ["client.mic_muted", "client.mic_unmuted"]) {
      expect(EVENT_TYPES as readonly string[]).not.toContain(clientType);
    }
  });
});
