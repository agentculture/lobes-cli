/**
 * Fixture events for the live event-stream log (issue #151 t12).
 *
 * This is the "documentation of the wire" deliverable: one representative
 * payload for EVERY `EventType` and EVERY `ErrorCode` in
 * `realtime-events.ts`, shaped exactly like `event_to_dict()` in
 * `lobes/realtime/_session.py` would serialize it. It backs three things:
 *
 *   1. `event-log.test.ts`'s coverage tests — they iterate
 *      `EVENT_TYPES`/`ERROR_CODES` and fail if this file stops covering one
 *      (e.g. because a new error code landed server-side and nobody added
 *      a fixture for it).
 *   2. `dev-events.astro` — a server-free page that replays this exact
 *      story through the real `EventStream` component, so the whole UI is
 *      reviewable with `npm run dev` alone.
 *   3. Anyone reading the wire format for the first time: this file IS the
 *      shape, not a description of it.
 *
 * The story is one session, three user turns:
 *   - turn 1 completes normally (a short back-and-forth about Mars).
 *   - turn 2 runs long enough to hit `VAD_MAX_TURN_MS` and force-commits
 *     (`reason: "max_turn"`, not `"silence"` — the two committed-turn
 *     reasons a VAD-tuning operator needs to tell apart), and its reply is
 *     cut short by a barge-in (`response.interrupted`).
 *   - turn 3 is deliberately ears-only: no response is triggered, proving
 *     the default (non-conversational) path renders fine too.
 * Then every named error code is played back once, standalone, followed by
 * `session.closed`.
 *
 * `at_ms` is present on every boundary event here on purpose — see
 * `realtime-events.ts`'s module doc for why the LIVE server does not send
 * it yet (app.py doesn't thread `_segmenter.py`'s `at_ms` through
 * `Session.begin_speech`/`end_speech`). These fixtures render the UI's
 * target state; `event-log.test.ts` separately covers the current,
 * `at_ms`-less fallback so both paths stay proven.
 */

import type { RawEvent, ConnectionState } from "./realtime-events";

const SESSION_ID = "sess_a1b2c3d4e5f6a7b8c9d0e1f2";
const T0 = 1_000_000; // an arbitrary monotonic-clock base, matching timestamp_ms()'s domain

function ev(offsetMs: number, fields: Record<string, unknown>): RawEvent {
  return {
    session_id: SESSION_ID,
    event_id: `event_${Math.random().toString(36).slice(2, 10)}`,
    timestamp_ms: T0 + offsetMs,
    ...fields,
  } as RawEvent;
}

const ITEM_1 = "item_0001aaaaaaaaaaaaaaaaaaaa";
const ITEM_2 = "item_0002bbbbbbbbbbbbbbbbbbbb";
const ITEM_3 = "item_0003cccccccccccccccccccc";
const ITEM_4 = "item_0004dddddddddddddddddddd";
const RESP_1 = "resp_0001eeeeeeeeeeeeeeeeeeee";
const RESP_2 = "resp_0002ffffffffffffffffffff";

// A tiny valid base64 payload stands in for real PCM16 — the log only
// counts/coalesces deltas, it never decodes audio (see event-log.ts).
const STUB_DELTA = "UklGRiQAAABXQVZFZm10IBAAAAABAAEA";

export const EVENT_FIXTURES: RawEvent[] = [
  ev(0, {
    type: "session.created",
    config: {
      input_audio_format: "pcm16",
      input_sample_rate: 24000,
      channels: 1,
      turn_detection: "server_vad",
      aec_mode: "none",
      system_prompt: null,
    },
  }),

  // --- turn 1: completes normally ---
  ev(400, { type: "input_audio_buffer.speech_started", item_id: ITEM_1, at_ms: 128 }),
  ev(2400, {
    type: "input_audio_buffer.speech_stopped",
    item_id: ITEM_1,
    at_ms: 2048,
    reason: "silence",
  }),
  ev(2900, {
    type: "conversation.item.input_audio_transcription.completed",
    item_id: ITEM_1,
    text: "What's the weather like on Mars?",
  }),
  ev(3000, { type: "response.created", response_id: RESP_1, item_id: ITEM_1 }),
  ev(4200, {
    type: "response.text.done",
    response_id: RESP_1,
    text: "Mars averages about minus sixty degrees Celsius, with dust storms that can last for weeks.",
  }),
  ev(4300, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4400, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4500, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4600, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4900, { type: "response.done", response_id: RESP_1 }),

  // --- turn 2: runs long enough to force-commit, then gets interrupted ---
  ev(10000, { type: "input_audio_buffer.speech_started", item_id: ITEM_2, at_ms: 9000 }),
  ev(40000, {
    type: "input_audio_buffer.speech_stopped",
    item_id: ITEM_2,
    at_ms: 39000,
    reason: "max_turn",
  }),
  ev(40500, {
    type: "conversation.item.input_audio_transcription.completed",
    item_id: ITEM_2,
    text: "okay actually never mind, tell me about Io instead",
  }),
  ev(40600, { type: "response.created", response_id: RESP_2, item_id: ITEM_2 }),
  ev(41800, {
    type: "response.text.done",
    response_id: RESP_2,
    text: "Io is the most volcanically active body in the solar system—",
  }),
  ev(41900, { type: "response.audio.delta", response_id: RESP_2, item_id: ITEM_2, delta: STUB_DELTA }),
  ev(42000, { type: "response.audio.delta", response_id: RESP_2, item_id: ITEM_2, delta: STUB_DELTA }),
  ev(42300, { type: "response.interrupted", response_id: RESP_2, truncated: true }),

  // the barge-in's own onset, then a third turn that never triggers a
  // response — proving the ears-only default still renders cleanly
  ev(42320, { type: "input_audio_buffer.speech_started", item_id: ITEM_3, at_ms: 41200 }),
  ev(43800, {
    type: "input_audio_buffer.speech_stopped",
    item_id: ITEM_3,
    at_ms: 42700,
    reason: "silence",
  }),
  ev(44200, {
    type: "conversation.item.input_audio_transcription.completed",
    item_id: ITEM_3,
    text: "never mind, forget it",
  }),

  // --- every named error code, once each ---
  ev(50000, {
    type: "error",
    code: "invalid_session_config",
    message: "unsupported input_sample_rate 8000; accepted rates are (24000, 16000)",
  }),
  ev(50100, {
    type: "error",
    code: "vad_unavailable",
    message: "RuntimeError: failed to load silero_vad from torch.hub",
  }),
  ev(50150, {
    type: "error",
    code: "invalid_wire_event",
    message:
      "invalid_append_event: 'input_audio_buffer.append' requires a base64 string 'audio' field, got None",
  }),
  ev(50200, {
    type: "error",
    code: "stt_forward_failed",
    item_id: ITEM_4,
    message: "POST http://stt:8090/v1/audio/transcriptions -> 503 Service Unavailable",
  }),
  ev(50300, {
    type: "error",
    code: "generate_failed",
    message: "POST http://gateway:8000/v1/chat/completions -> 404 role_infeasible (hosted_by=thor)",
  }),
  ev(50400, {
    type: "error",
    code: "tts_failed",
    message: "POST http://tts:8091/v1/audio/speech -> 500 Internal Server Error",
  }),
  ev(50500, {
    type: "error",
    code: "response_timeout",
    message: "tts stage exceeded 60000ms",
  }),

  ev(51000, { type: "session.closed", reason: "client_disconnect" }),
];

/** Every `ConnectionState` transition worth demonstrating, in order. */
export const CONNECTION_FIXTURES: { state: ConnectionState; detail?: string }[] = [
  { state: "connecting", detail: "opening ws://localhost:4321/v1/realtime" },
  { state: "connected" },
  { state: "disconnected", detail: "server closed the connection (1000)" },
];

/** A second, standalone scenario: a transport error, never a named server error. */
export const CONNECTION_ERROR_FIXTURE: { state: ConnectionState; detail?: string }[] = [
  { state: "connecting", detail: "opening ws://localhost:4321/v1/realtime" },
  { state: "error", detail: "WebSocket error — is the local proxy running?" },
];

export interface ReplayHandle {
  cancel(): void;
}

export interface ReplayTarget {
  pushEvent(raw: unknown): unknown;
  setConnectionState(state: ConnectionState, detail?: string): void;
}

/**
 * Replay `EVENT_FIXTURES` (and, first, `CONNECTION_FIXTURES`) against
 * *target* with a small stagger between each, so a human watching the dev
 * page sees the log fill in roughly the way a live session would rather
 * than all at once. Returns a handle to cancel an in-flight replay (e.g.
 * before unmount, or when the "replay again" control is pressed mid-run).
 *
 * `stepMs` defaults small (40ms) — this is playback PACING for a human
 * demo, not a CSS animation, so it is not gated behind
 * prefers-reduced-motion (nothing here moves on screen by itself; each
 * step is a discrete row appearing, same as a real message arriving).
 */
export function replayFixtures(target: ReplayTarget, stepMs = 40): ReplayHandle {
  let cancelled = false;
  const timers: ReturnType<typeof setTimeout>[] = [];

  let step = 0;
  const schedule = (fn: () => void) => {
    const t = setTimeout(() => {
      if (!cancelled) fn();
    }, step * stepMs);
    timers.push(t);
    step += 1;
  };

  for (const transition of CONNECTION_FIXTURES) {
    schedule(() => target.setConnectionState(transition.state, transition.detail));
  }
  for (const event of EVENT_FIXTURES) {
    schedule(() => target.pushEvent(event));
  }

  return {
    cancel() {
      cancelled = true;
      timers.forEach(clearTimeout);
    },
  };
}
