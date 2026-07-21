/**
 * The /v1/realtime wire vocabulary — issue #151 t12.
 *
 * This module is the "documentation of the wire" the task brief asked for:
 * a hand-mirrored copy of `lobes/realtime/_session.py`'s `EventType` and
 * `ErrorCode` enums, plus the display metadata (label / short badge / icon
 * / colour family) the UI uses to render every event type and every named
 * error code distinctly. `site/` has no Python import path into `lobes/`,
 * so this file is the single place that vocabulary is re-declared for the
 * browser — keep it in sync BY HAND with:
 *
 *   - lobes/realtime/_session.py   (EventType, ErrorCode)
 *   - lobes/realtime/_segmenter.py (SpeechStarted/SpeechStopped.at_ms, the
 *     quantised-audio-stream-time field boundary events carry — see the
 *     `at_ms` handling note below)
 *
 * `event-log.test.ts`'s "fixture coverage" tests fail loudly if
 * `EVENT_TYPES`/`ERROR_CODES` grows a member `event-fixtures.ts` does not
 * exercise — so an out-of-sync mirror is caught in CI, not discovered live.
 *
 * `at_ms` — read this before touching boundary-event rendering
 * -----------------------------------------------------------------------
 * `_segmenter.py`'s `SpeechStarted`/`SpeechStopped` carry `at_ms`: elapsed
 * *audio-stream* time, quantised to whole 32ms VAD chunks — NOT wall-clock,
 * and deliberately so (it is what makes VAD_THRESHOLD/VAD_SILENCE_MS/
 * VAD_PREFIX_PADDING_MS tuning observable: the gap between a
 * speech_started.at_ms and the next speech_stopped.at_ms is exactly the
 * turn length the segmenter measured). As of this task, `app.py`'s
 * `_emit_turn_events` calls `session.begin_speech()`/`session.end_speech()`
 * with NO arguments, and `_session.py`'s `SpeechStartedEvent`/
 * `SpeechStoppedEvent` dataclasses carry only `timestamp_ms` (a
 * `time.monotonic()`-based process clock, not even wall-clock) — so
 * `at_ms` does not yet reach the wire. This module and `event-log.ts`
 * treat `at_ms` as an OPTIONAL field on boundary events on purpose: reading
 * it when present costs nothing, and the moment a follow-up threads the
 * segmenter's `at_ms` (and its `reason`: "silence" vs "max_turn") through
 * `Session.begin_speech`/`end_speech` into the wire event, this UI starts
 * rendering the real value with no further change. Until then, boundary
 * rows fall back to elapsed time since `session.created` computed from
 * `timestamp_ms` — labelled "elapsed" and explicitly NOT called
 * "audio-stream time", because it isn't one (see `formatBoundaryTiming`
 * in `event-log.ts`). Rendering that fallback as if it were `at_ms` would
 * be exactly the lie the task brief warned against.
 */

// ---------------------------------------------------------------------------
// EventType — mirrors _session.py's EventType enum, in the same order.
// ---------------------------------------------------------------------------

export const EVENT_TYPES = [
  "session.created",
  "session.closed",
  "input_audio_buffer.speech_started",
  "input_audio_buffer.speech_stopped",
  "conversation.item.input_audio_transcription.completed",
  "error",
  "response.created",
  "response.text.done",
  "response.audio.delta",
  "response.done",
  "response.interrupted",
] as const;

export type EventType = (typeof EVENT_TYPES)[number];

// ---------------------------------------------------------------------------
// CLIENT-ORIGIN events — deliberately NOT part of EVENT_TYPES.
//
// EVENT_TYPES above mirrors _session.py exactly, and that mirror is what
// keeps this file honest; folding browser-local events into it would make it
// claim the server sends things it does not. These are emitted by the mic
// island (t18) so the operator can tell "muted" from "silence" (which emits
// nothing at all) and from "disconnected" — the three states that would
// otherwise look identical in a quiet log. They carry `origin: "client"` on
// the wire-shaped payload for the same reason.
//
// They exist because of approved deviation d1: a USER-initiated mute is a
// control affordance, while an AUTOMATIC mute during playback stays
// forbidden (it is the AEC substitute that makes barge-in impossible). A
// muted row in the log is therefore never evidence of a bug — but a muted
// row the operator did not cause would be.
// ---------------------------------------------------------------------------

export const CLIENT_EVENT_TYPES = ["client.mic_muted", "client.mic_unmuted"] as const;

export type ClientEventType = (typeof CLIENT_EVENT_TYPES)[number];

// ---------------------------------------------------------------------------
// ErrorCode — mirrors _session.py's ErrorCode enum, in the same order.
// ---------------------------------------------------------------------------

export const ERROR_CODES = [
  "invalid_session_config",
  "vad_unavailable",
  "stt_forward_failed",
  "generate_failed",
  "tts_failed",
  "response_timeout",
] as const;

export type ErrorCode = (typeof ERROR_CODES)[number];

// ---------------------------------------------------------------------------
// Icon identifiers — see event-icons.ts for the actual SVG shapes. Every
// value here is a genuinely distinct silhouette, never a colour variant of
// another icon: colour is reinforcement, shape + the label/badge text below
// are the load-bearing signal (a colour-blind operator must still be able
// to tell every row apart).
// ---------------------------------------------------------------------------

export type IconId =
  | "session-open"
  | "session-close"
  | "boundary-start"
  | "boundary-stop"
  | "transcript"
  | "response-start"
  | "response-text"
  | "response-audio"
  | "response-done"
  | "response-interrupted"
  | "error-config"
  | "error-vad"
  | "error-stt"
  | "error-generate"
  | "error-tts"
  | "error-timeout"
  | "conn-connecting"
  | "conn-connected"
  | "conn-disconnected"
  | "conn-error"
  | "mic-muted"
  | "mic-unmuted"
  | "unknown";

/** Colour-role families — CSS classes, never the sole distinguishing signal. */
export type ColourFamily =
  | "lifecycle"
  | "boundary"
  | "transcript"
  | "response"
  | "interrupted"
  | "error"
  | "connection"
  | "client";

export interface EventKindMeta {
  /** Short, human label shown on the row. */
  label: string;
  icon: IconId;
  family: ColourFamily;
}

export const EVENT_KIND_META: Record<EventType, EventKindMeta> = {
  "session.created": { label: "session.created", icon: "session-open", family: "lifecycle" },
  "session.closed": { label: "session.closed", icon: "session-close", family: "lifecycle" },
  "input_audio_buffer.speech_started": {
    label: "speech_started",
    icon: "boundary-start",
    family: "boundary",
  },
  "input_audio_buffer.speech_stopped": {
    label: "speech_stopped",
    icon: "boundary-stop",
    family: "boundary",
  },
  "conversation.item.input_audio_transcription.completed": {
    label: "transcription.completed",
    icon: "transcript",
    family: "transcript",
  },
  error: { label: "error", icon: "unknown", family: "error" }, // overridden per-code, see ERROR_KIND_META
  "response.created": { label: "response.created", icon: "response-start", family: "response" },
  "response.text.done": {
    label: "response.text.done",
    icon: "response-text",
    family: "response",
  },
  "response.audio.delta": {
    label: "response.audio.delta",
    icon: "response-audio",
    family: "response",
  },
  "response.done": { label: "response.done", icon: "response-done", family: "response" },
  "response.interrupted": {
    label: "response.interrupted",
    icon: "response-interrupted",
    family: "interrupted",
  },
};

export interface ErrorKindMeta {
  /** 3-4 letter badge — the primary, colour-independent "which stage" signal. */
  badge: string;
  label: string;
  icon: IconId;
  /** One-line explanation of what this code means, shown as row detail. */
  hint: string;
}

export const ERROR_KIND_META: Record<ErrorCode, ErrorKindMeta> = {
  invalid_session_config: {
    badge: "CFG",
    label: "invalid_session_config",
    icon: "error-config",
    hint: "the session connect config was rejected before any audio was accepted",
  },
  vad_unavailable: {
    badge: "VAD",
    label: "vad_unavailable",
    icon: "error-vad",
    hint: "Silero failed to load or run — distinct from silence, which emits no event at all",
  },
  stt_forward_failed: {
    badge: "STT",
    label: "stt_forward_failed",
    icon: "error-stt",
    hint: "a committed turn's forward to Parakeet (speech-to-text) failed",
  },
  generate_failed: {
    badge: "GEN",
    label: "generate_failed",
    icon: "error-generate",
    hint: "the reply's forward to the generate lane failed (including a role_infeasible 404)",
  },
  tts_failed: {
    badge: "TTS",
    label: "tts_failed",
    icon: "error-tts",
    hint: "the reply's forward to the text-to-speech lane failed",
  },
  response_timeout: {
    badge: "TMO",
    label: "response_timeout",
    icon: "error-timeout",
    hint: "a response stage exceeded its deadline — see the message for which stage",
  },
};

// ---------------------------------------------------------------------------
// Client-side connection state — NOT a server wire event. The WebSocket
// transport lifecycle (owned by whichever island drives the socket, e.g.
// t13's proxy/connection controls) is a separate concern from the named
// `error` event family on purpose: a dropped connection is not a server
// error, and the two must never be rendered as if they were the same thing
// (task acceptance: "a named error vs. a disconnect").
// ---------------------------------------------------------------------------

export const CONNECTION_STATES = [
  "idle",
  "connecting",
  "connected",
  "disconnected",
  "error",
] as const;

export type ConnectionState = (typeof CONNECTION_STATES)[number];

export interface ConnectionKindMeta {
  label: string;
  icon: IconId;
}

export const CONNECTION_KIND_META: Record<ConnectionState, ConnectionKindMeta> = {
  idle: { label: "Not connected", icon: "conn-disconnected" },
  connecting: { label: "Connecting…", icon: "conn-connecting" },
  connected: { label: "Connected", icon: "conn-connected" },
  disconnected: { label: "Disconnected", icon: "conn-disconnected" },
  error: { label: "Connection error", icon: "conn-error" },
};

/** A loosely-typed raw wire event — client-received JSON is untrusted input. */
export interface RawEvent {
  type: string;
  session_id?: string;
  event_id?: string;
  timestamp_ms?: number;
  [key: string]: unknown;
}

export function isKnownEventType(type: string): type is EventType {
  return (EVENT_TYPES as readonly string[]).includes(type);
}

export function isClientEventType(type: string): type is ClientEventType {
  return (CLIENT_EVENT_TYPES as readonly string[]).includes(type);
}

/** Display metadata for the client-origin events, kept beside the server map. */
export const CLIENT_EVENT_KIND_META: Record<ClientEventType, EventKindMeta> = {
  "client.mic_muted": { label: "mic muted (you)", icon: "mic-muted", family: "client" },
  "client.mic_unmuted": { label: "mic unmuted (you)", icon: "mic-unmuted", family: "client" },
};

export function isKnownErrorCode(code: string): code is ErrorCode {
  return (ERROR_CODES as readonly string[]).includes(code);
}
