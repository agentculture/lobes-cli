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
 * The story is one session, in Hebrew, with a declared demo tool, across
 * four user turns:
 *   - `session.updated` answers a `session.update` declaring one demo tool
 *     (`get_current_time`) right after `session.created` — the
 *     hebrew-realtime t4 shape this file also documents.
 *   - turn 1 completes normally (a short back-and-forth about Mars, in
 *     Hebrew — proving the RTL rendering path against real fixture text),
 *     and its `response.done` carries a `timings` mapping (no `tool_wait`:
 *     this turn called no tool).
 *   - turn 1b is a full tool round trip: `response.function_call_arguments.done`
 *     arrives mid-response, and the follow-up `response.done` carries a
 *     `tool_wait` stage.
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
const ITEM_5 = "item_0005eeeeeeeeeeeeeeeeeeee";
const ITEM_6 = "item_0006ffffffffffffffffffff";
const RESP_1 = "resp_0001eeeeeeeeeeeeeeeeeeee";
const RESP_2 = "resp_0002ffffffffffffffffffff";
const RESP_3 = "resp_0003aaaaaaaaaaaaaaaaaaaa";
const RESP_4 = "resp_0004bbbbbbbbbbbbbbbbbbbb";

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
      language: "he",
    },
  }),

  // The client's session.update (declaring one demo tool) answered —
  // hebrew-realtime t4. Echoes ONLY the fields that took effect.
  ev(50, {
    type: "session.updated",
    session: {
      tools: [
        {
          type: "function",
          name: "get_current_time",
          description: "Return the current local time and weekday.",
          parameters: { type: "object", properties: {} },
        },
      ],
      tool_choice: "auto",
      language: "he",
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
    // Hebrew, on purpose — proving the RTL rendering path with real fixture
    // data rather than only a synthetic test string: "what is the weather
    // like on Mars?"
    text: "מה מזג האוויר במאדים?",
  }),
  ev(3000, { type: "response.created", response_id: RESP_1, item_id: ITEM_1 }),
  ev(4200, {
    type: "response.text.done",
    response_id: RESP_1,
    text: "בממוצע כשישים מעלות מתחת לאפס, עם סופות אבק שיכולות להימשך שבועות.",
  }),
  ev(4300, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4400, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4500, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4600, { type: "response.audio.delta", response_id: RESP_1, item_id: ITEM_1, delta: STUB_DELTA }),
  ev(4900, {
    type: "response.done",
    response_id: RESP_1,
    // Additive and optional (StageTimings) — an unmeasured stage (no tool
    // call on this turn) is ABSENT, never zeroed. "phonikud" is the Hebrew
    // niqqud-restoration stage ahead of TTS; an English deployment never
    // reports it.
    timings: { stt: 410, generate: 780, phonikud: 90, tts: 260, first_delta: 1180 },
  }),

  // --- turn 1b: a tool round trip, in Hebrew (hebrew-realtime t14/t17 shape) ---
  ev(5200, { type: "input_audio_buffer.speech_started", item_id: ITEM_5, at_ms: 5000 }),
  ev(6800, {
    type: "input_audio_buffer.speech_stopped",
    item_id: ITEM_5,
    at_ms: 6600,
    reason: "silence",
  }),
  ev(6900, {
    type: "conversation.item.input_audio_transcription.completed",
    item_id: ITEM_5,
    text: "מה השעה עכשיו?", // "what time is it now?"
  }),
  ev(7000, { type: "response.created", response_id: RESP_3, item_id: ITEM_5 }),
  ev(7300, {
    type: "response.function_call_arguments.done",
    response_id: RESP_3,
    item_id: ITEM_5,
    call_id: "call_0001aaaaaaaaaaaaaaaaaaaa",
    name: "get_current_time",
    arguments: "{}",
  }),
  ev(7900, {
    type: "response.text.done",
    response_id: RESP_3,
    text: "השעה עכשיו 14:32, יום שלישי.",
  }),
  ev(8000, { type: "response.audio.delta", response_id: RESP_3, item_id: ITEM_5, delta: STUB_DELTA }),
  ev(8300, {
    type: "response.done",
    response_id: RESP_3,
    timings: { stt: 210, generate: 640, tool_wait: 180, phonikud: 70, tts: 240, first_delta: 900 },
  }),

  // --- turn 1c: an INTERRUPTED tool turn — the model called a tool and a
  // barge-in landed before the client's function_call_output ever went out.
  // Server-side this is exactly what leaves a call CLOSED (see
  // lobes/realtime/_conversation.py's TOOL_OUTPUT_CALL_CLOSED): a late
  // result would be refused as invalid_wire_event with reason "call_closed".
  // This fixture only shows the half a browser observes without sending
  // anything back — the UI must not assume a function_call_output always
  // follows a tool call.
  ev(8600, { type: "input_audio_buffer.speech_started", item_id: ITEM_6, at_ms: 8500 }),
  ev(8900, {
    type: "input_audio_buffer.speech_stopped",
    item_id: ITEM_6,
    at_ms: 8800,
    reason: "silence",
  }),
  ev(8950, {
    type: "conversation.item.input_audio_transcription.completed",
    item_id: ITEM_6,
    text: "וגם תגלגל קובייה", // "and also roll a die"
  }),
  ev(9000, { type: "response.created", response_id: RESP_4, item_id: ITEM_6 }),
  ev(9200, {
    type: "response.function_call_arguments.done",
    response_id: RESP_4,
    item_id: ITEM_6,
    call_id: "call_0002bbbbbbbbbbbbbbbbbbbb",
    name: "roll_dice",
    arguments: '{"sides": 6}',
  }),
  ev(9350, { type: "response.interrupted", response_id: RESP_4, truncated: true }),

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
