/*
 * mic-island.ts — the mic + playback island (#151 t11).
 *
 * Builds its own DOM, owns the one AudioContext both halves share, wires
 * `MicCapture` to `DeltaPlayer`, and renders a state that is honest about
 * which of several very different silences you are looking at.
 *
 * ── One gesture arms both halves ────────────────────────────────────────
 *
 * The AudioContext is constructed inside the start button's click handler,
 * never at mount: an AudioContext created outside a gesture starts suspended
 * (a browser rule), and the microphone prompt is a human decision that should
 * never fire because a page loaded. Nothing here touches
 * `navigator.mediaDevices` or Web Audio until that click, and the tests prove
 * it by counting constructor calls.
 *
 * ── Wiring: two DOM CustomEvents, or two function calls ─────────────────
 *
 * This island does not own the WebSocket — a sibling task does. The seam
 * between them is deliberately the dumbest possible one, so neither side has
 * to import the other:
 *
 *   OUT  every encoded frame is dispatched as `lobes:client-event` on
 *        `document`, with the finished `input_audio_buffer.append` event as
 *        `detail`. Whoever holds the socket forwards `detail` verbatim.
 *   IN   every server event is expected as `lobes:server-event` on
 *        `document`, with the parsed event object as `detail`.
 *
 * Both names are options, and both directions have a direct-call equivalent
 * ({@link MicIslandHandle.setSender}, {@link MicIslandHandle.handleServerEvent})
 * for a connection layer that would rather call a function than dispatch an
 * event. See the mounting notes in `MicIsland.astro`.
 *
 * ── What the states are for ─────────────────────────────────────────────
 *
 * A mic that hears nothing, a mic the browser blocked, a mic that does not
 * exist, a page served from an origin where capture is impossible, and a
 * server that returned a named error are five different situations with five
 * different next steps. They render as five different things — distinct
 * `data-state` values, distinct colours, distinct words — because collapsing
 * them into "no audio" is exactly the failure this site exists to prevent.
 *
 * ── `muted` — a deliberate, user-triggered exception (issue #151 t18, d1) ──
 *
 * t11 shipped this file next to `mic-capture.ts`'s claim that there is no
 * state in which the mic is open but not listening, because that state would
 * be the half-duplex mute `scripts/realtime-voice-loop.py` uses to survive
 * without echo cancellation — and barge-in cannot coexist with a mic that
 * stops listening while the machine talks. That claim was true when there
 * was no way to own AEC except by muting.
 *
 * Deviation d1 (approved 2026-07-21, recorded in the plan under issue #151
 * t18) narrows it, it does not repeal it: real hardware now owns echo
 * cancellation at the client edge (Reachy's firmware, this browser's
 * `echoCancellation` constraint), so a mic that is open-but-not-relaying is
 * no longer automatically the AEC-substitute hack — PROVIDED a human, not a
 * playback or response event, put it there. `muted` is exactly that: the
 * device stays held (`MicCapture` keeps capturing, the worklet keeps
 * running, the browser's recording indicator stays lit) and only the
 * OUTBOUND relay in this file — `emitClientEvent`'s caller, below — is
 * gated. Nothing about `mic-capture.ts` changed: it still never touches
 * `track.enabled`, a gain node, or any element's `.muted`; the gate lives one
 * layer up, in the part of the system a human's own click can reach.
 *
 * The constraint that survives d1, unnarrowed: nothing may flip that gate
 * automatically. `handleServerEvent`, `handlePlaybackStop`, and
 * `notifyDisconnected` — every function that reacts to something the SERVER
 * or the CONNECTION did — are wrapped in `AUTOMATIC-MUTE-FORBIDDEN-ZONE`
 * markers that `no-mic-mute.test.ts` scans, so an attempt to wire muting to
 * playback slides right back into the failure d1 exists to keep out.
 */

import type { BrowserAudioDeps, AudioContextLike } from "./audio-graph";
import { browserAudioDeps } from "./audio-graph";
import type { MicCaptureState, MicStateDetail } from "./mic-capture";
import { MicCapture } from "./mic-capture";
import type { PlaybackStopInfo } from "./audio-playback";
import { DeltaPlayer } from "./audio-playback";
import type { AppendEvent } from "./pcm-wire";
import { DEFAULT_INPUT_SAMPLE_RATE, TTS_OUTPUT_SAMPLE_RATE } from "./pcm-wire";

/** Default name of the DOM event carrying an outbound append event. */
export const CLIENT_EVENT_NAME = "lobes:client-event";

/** Default name of the DOM event carrying an inbound server event. */
export const SERVER_EVENT_NAME = "lobes:server-event";

/** Dispatched once at mount so a connection layer can read the negotiated input rate. */
export const MIC_READY_EVENT_NAME = "lobes:mic-ready";

/**
 * What the island is doing, as one word.
 *
 * `listening` and `speaking` are both "the mic is open, and sent" — the
 * difference is whether a reply is playing over it. `muted` is the one
 * exception carved out by deviation d1 (see the module doc above): the mic
 * is open and the device is held exactly like `listening`/`speaking`, but
 * nothing captured is relayed outward. It renders as its own word and its
 * own glyph precisely so it is never mistaken for either of them, for
 * silence, or for `stopped` (the device released outright).
 */
export type MicIslandState =
  | "idle"
  | "arming"
  | "listening"
  | "speaking"
  | "muted"
  | "stopped"
  | "denied"
  | "no-device"
  | "unsupported"
  | "failed";

interface StateCopy {
  label: string;
  message: string;
}

interface HintLocation {
  hostname: string;
  port: string;
  protocol?: string;
}

/** The lowest port a non-root user can bind locally. */
const FIRST_UNPRIVILEGED_PORT = 1024;
/** Where the harness normally lives, and the local end of a privileged forward. */
const DEFAULT_HARNESS_PORT = "4321";

/**
 * The port the page is really served on. `location.port` is EMPTY on a default
 * port, so a naive `|| fallback` invents a port the page was never on and emits
 * a forwarding command for the wrong service.
 */
function effectivePort(loc: HintLocation): string {
  if (loc.port) return loc.port;
  return loc.protocol === "https:" ? "443" : "80";
}

/**
 * A copy-pasteable remedy for an insecure origin, derived from the current URL.
 *
 * `getUserMedia` gives a page NO microphone outside a secure context, and the
 * symptom points anywhere but the address bar — so the message names the exact
 * URL and the exact forwarding command for THIS page rather than describing the
 * rule and leaving the reader to apply it.
 *
 * Only call this once the origin is KNOWN to be the problem — see
 * `unsupportedMessage`, which decides that. Asserting it blind is how a missing
 * `AudioWorkletNode` ends up sending someone to fix their address bar.
 */
export function secureContextHint(location?: HintLocation): string {
  const loc =
    location ??
    (typeof window !== "undefined"
      ? {
          hostname: window.location.hostname,
          port: window.location.port,
          protocol: window.location.protocol,
        }
      : { hostname: "localhost", port: DEFAULT_HARNESS_PORT, protocol: "http:" });
  const base =
    "This page cannot capture audio from this origin. getUserMedia needs a secure context — https:// or a localhost origin, and nothing else.";
  if (loc.hostname === "localhost" || loc.hostname === "127.0.0.1" || loc.hostname === "::1") {
    // Already on localhost: the origin is not the problem, so do not send the
    // reader chasing it. Something else removed the API.
    return `${base} This page IS on localhost, so the origin is not the problem — the browser or an extension has removed getUserMedia.`;
  }
  const remotePort = effectivePort(loc);
  // Binding a privileged port locally needs root, so a page served on 80/443
  // gets forwarded to the harness port instead of a command that would fail.
  const localPort =
    Number(remotePort) < FIRST_UNPRIVILEGED_PORT ? DEFAULT_HARNESS_PORT : remotePort;
  return `${base} From another machine: ssh -L ${localPort}:localhost:${remotePort} ${loc.hostname} — then open http://localhost:${localPort}/ . On this box: open http://localhost:${remotePort}/ directly.`;
}

const MISSING_AUDIO_APIS =
  "This browser is missing the Web Audio APIs this page needs — getUserMedia, AudioContext, or AudioWorkletNode. Try a current Chrome or Firefox, and check whether an extension or a hardened privacy setting has removed them.";

/**
 * Both causes named, neither asserted — for when `isSecureContext` cannot be
 * read, and as the static `STATE_COPY` fallback.
 */
export const UNSUPPORTED_CAUSE_UNKNOWN = `${MISSING_AUDIO_APIS} If this page is served over plain http:// from another machine, that alone also removes getUserMedia — reach it over localhost or https:// instead.`;

/**
 * The message for the `unsupported` state, which has TWO unrelated causes.
 *
 * An insecure origin strips `navigator.mediaDevices` in most browsers, so it
 * lands in the same capability check as a genuinely missing `AudioWorkletNode`.
 * Blaming the origin unconditionally sends someone with an old browser to go
 * set up SSH forwarding that will change nothing — so read `isSecureContext`
 * and only assert a cause when it is actually known.
 */
export function unsupportedMessage(env?: {
  isSecureContext?: boolean;
  location?: HintLocation;
}): string {
  const secure =
    env?.isSecureContext ??
    (typeof globalThis !== "undefined" ? globalThis.isSecureContext : undefined);
  if (secure === false) return secureContextHint(env?.location);
  if (secure === true) return MISSING_AUDIO_APIS;
  // No isSecureContext to read at all. Either cause is live, so name both
  // rather than pick one and send half the readers somewhere useless.
  return UNSUPPORTED_CAUSE_UNKNOWN;
}

const STATE_COPY: Record<MicIslandState, StateCopy> = {
  idle: {
    label: "Not armed",
    message: "Nothing is captured and nothing plays until you start.",
  },
  arming: {
    label: "Arming",
    message: "Asking the browser for the microphone…",
  },
  listening: {
    label: "Listening",
    message: "Mic open. Speak — the server's VAD decides where your turn ends.",
  },
  speaking: {
    label: "Reply playing",
    message: "The mic is still open while lobes speaks. Talk over it to interrupt.",
  },
  muted: {
    label: "Muted",
    message:
      "Mic held open, device still yours — nothing captured is leaving this tab. Reply audio, if any, keeps playing; muting never interrupts it. Press Unmute to resume sending.",
  },
  stopped: {
    label: "Mic off",
    message:
      "The microphone is released — the device is fully let go and playback is cleared. Press start to re-arm; that takes the same gesture path as the very first start.",
  },
  denied: {
    label: "Mic blocked",
    message:
      "The browser refused microphone access. Allow the mic for this origin in the site settings, then start again.",
  },
  "no-device": {
    label: "No microphone",
    message:
      "The browser found no audio input device. This is not a permission problem — and a mic being physically plugged in is not enough: the OS has to expose it as an INPUT. On PipeWire/PulseAudio a card can sit in an output-only profile, which shows a working device to `arecord` and nothing at all to the browser. Check `pactl list sources short` for a non-monitor source; if there is none, switch the card to a duplex profile (`pactl set-card-profile <card> output:analog-stereo+input:stereo-fallback`) and start again.",
  },
  unsupported: {
    label: "Capture unavailable",
    // A cause-neutral fallback ONLY. The real message is resolved per-render by
    // `unsupportedMessage()` and passed through `detail`, because it depends on
    // `isSecureContext` and the live URL — neither of which is knowable here at
    // module-init time (this module is also evaluated during the Astro build,
    // where there is no `window` to read).
    message: UNSUPPORTED_CAUSE_UNKNOWN,
  },
  failed: {
    label: "Capture failed",
    message: "The microphone could not be started.",
  },
};

export interface MicIslandOptions {
  /** The rate to REQUEST for `input_sample_rate` (24000 or 16000). */
  inputSampleRate?: number;
  /** Where client/server DOM events are dispatched and listened for. Defaults to `document`. */
  eventTarget?: EventTarget;
  clientEventName?: string;
  serverEventName?: string;
  /** A direct outbound sink, called in addition to the DOM event. */
  send?: (event: AppendEvent) => void;
  appendFrameMs?: number;
  workletUrl?: string;
  deps?: Partial<BrowserAudioDeps>;
}

export interface MicIslandHandle {
  readonly root: HTMLElement;
  getState(): MicIslandState;
  /** The rate outbound audio is encoded at — put this in the connect URL. */
  getInputSampleRate(): number;
  start(): Promise<boolean>;
  stop(): Promise<void>;
  /** Whether outbound frames are currently withheld (issue #151 t18, d1). */
  isMuted(): boolean;
  /**
   * Set mute directly — the function-call equivalent of clicking the mute
   * button. A no-op while the device is not held (nothing to mute) or when
   * already at the requested value. Never call this from code that reacts
   * to a server or connection event: see the module doc's d1 section and
   * `no-mic-mute.test.ts`.
   */
  setMuted(muted: boolean): void;
  /** Install (or clear) a direct sink for outbound append events. */
  setSender(send: ((event: AppendEvent) => void) | null): void;
  /** Feed one server event in. Accepts the parsed object or its JSON text. */
  handleServerEvent(event: unknown): void;
  /** The socket went away: stop playback, release the mic, say so. */
  notifyDisconnected(reason?: string): void;
  destroy(): void;
}

interface ServerEventLike {
  type?: unknown;
  delta?: unknown;
  code?: unknown;
  message?: unknown;
  reason?: unknown;
  config?: { input_sample_rate?: unknown } | unknown;
}

function parseServerEvent(raw: unknown): ServerEventLike | null {
  let value = raw;
  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return null;
    }
  }
  if (typeof value === "object" && value !== null) return value as ServerEventLike;
  return null;
}

function readConfiguredRate(event: ServerEventLike): number | null {
  const config = event.config;
  if (typeof config !== "object" || config === null) return null;
  const rate = (config as { input_sample_rate?: unknown }).input_sample_rate;
  const parsed = typeof rate === "string" ? Number(rate) : rate;
  return typeof parsed === "number" && Number.isFinite(parsed) ? parsed : null;
}

function seconds(ms: number): string {
  return `${(ms / 1000).toFixed(2)} s`;
}

function el<K extends keyof HTMLElementTagNameMap>(
  doc: Document,
  tag: K,
  className?: string,
): HTMLElementTagNameMap[K] {
  const node = doc.createElement(tag);
  if (className) node.className = className;
  return node;
}

/**
 * Build the island's DOM and wire it up.
 *
 * Idempotent per root: mounting twice returns the first handle rather than
 * stacking two microphones on one element.
 */
export function mountMicIsland(root: HTMLElement, options: MicIslandOptions = {}): MicIslandHandle {
  const existing = MOUNTED.get(root);
  if (existing) return existing;

  const doc = root.ownerDocument;
  const target = options.eventTarget ?? doc;
  const clientEventName = options.clientEventName ?? CLIENT_EVENT_NAME;
  const serverEventName = options.serverEventName ?? SERVER_EVENT_NAME;
  const deps: BrowserAudioDeps = { ...browserAudioDeps, ...(options.deps ?? {}) };

  // ---------------------------------------------------------------- markup
  root.classList.add("mic-island");
  root.replaceChildren();

  const topRow = el(doc, "div", "mic-row");
  const chip = el(doc, "span", "mic-chip");
  const dot = el(doc, "span", "mic-dot");
  dot.setAttribute("aria-hidden", "true");
  const chipLabel = el(doc, "span", "mic-chip-label");
  chip.append(dot, chipLabel);

  const startButton = el(doc, "button", "mic-button");
  startButton.type = "button";

  // The mute control (issue #151 t18, deviation d1). A second, independent
  // toggle from the start/stop control above: muting never releases the
  // device and never touches playback, so it gets its own button rather
  // than overloading the one that arms/disarms both halves.
  const muteButton = el(doc, "button", "mic-mute-button");
  muteButton.type = "button";
  muteButton.setAttribute("aria-pressed", "false");

  topRow.append(chip, startButton, muteButton);

  const message = el(doc, "p", "mic-message");
  message.setAttribute("role", "status");
  message.setAttribute("aria-live", "polite");

  const meterRow = el(doc, "div", "mic-meter-row");
  const meterLabel = el(doc, "span", "mic-meter-label");
  meterLabel.textContent = "input";
  const meter = el(doc, "div", "mic-meter");
  meter.setAttribute("role", "meter");
  meter.setAttribute("aria-label", "Microphone input level");
  meter.setAttribute("aria-valuemin", "0");
  meter.setAttribute("aria-valuemax", "100");
  meter.setAttribute("aria-valuenow", "0");
  const meterFill = el(doc, "span", "mic-meter-fill");
  meter.append(meterFill);
  const meterValue = el(doc, "span", "mic-meter-value");
  meterValue.textContent = "0%";
  meterRow.append(meterLabel, meter, meterValue);

  const facts = el(doc, "dl", "mic-facts");
  const factIn = el(doc, "dd");
  const factOut = el(doc, "dd");
  const factFrames = el(doc, "dd");
  // A separate counter from `factFrames`: captured and sent are the same
  // number until the human mutes, and deliberately different after — the
  // fact that used to be called "frames sent" is now honestly "frames
  // captured" (mic-capture.ts still encodes every frame regardless of
  // mute), and this new one is the honest "frames sent" — the count that
  // actually stalls the instant mute engages.
  const factSent = el(doc, "dd");
  for (const [term, value] of [
    ["mic →", factIn],
    ["reply ←", factOut],
    ["frames captured", factFrames],
    ["frames sent", factSent],
  ] as const) {
    const group = el(doc, "div", "mic-fact");
    const dt = el(doc, "dt");
    dt.textContent = term;
    group.append(dt, value);
    facts.append(group);
  }

  const playback = el(doc, "p", "mic-playback");
  const serverError = el(doc, "p", "mic-server-error");
  serverError.hidden = true;

  root.append(topRow, message, meterRow, facts, playback, serverError);

  // ----------------------------------------------------------------- state
  const requestedRate = options.inputSampleRate ?? DEFAULT_INPUT_SAMPLE_RATE;
  let state: MicIslandState = "idle";
  let detail: MicStateDetail = {};
  let context: AudioContextLike | null = null;
  let player: DeltaPlayer | null = null;
  let sender: ((event: AppendEvent) => void) | null = options.send ?? null;
  let droppedDeltas = 0;
  let lastMeterPercent = -1;
  let playbackNote = "Reply audio: nothing received yet.";
  let destroyed = false;
  // The mute gate (#151 t18, d1). Only ever flipped from `toggleMute`, which
  // only ever runs from the mute button's click handler — never from inside
  // an AUTOMATIC-MUTE-FORBIDDEN-ZONE. See the module doc above.
  let muted = false;
  let framesForwarded = 0;

  const capture = new MicCapture({
    inputSampleRate: requestedRate,
    appendFrameMs: options.appendFrameMs,
    workletUrl: options.workletUrl,
    deps,
    onAppend: (event) => {
      // mic-capture.ts encodes and counts this frame unconditionally — the
      // device and the worklet do not know or care whether the human muted
      // anything (see mic-capture.ts's own "two things this module will
      // never do"). The gate is here, in the only layer a click handler can
      // reach: a muted frame is captured (the fact below still climbs) and
      // simply never forwarded (the OTHER fact does not).
      factFrames.textContent = String(capture.framesSent);
      if (muted) return;
      emitClientEvent(event);
      framesForwarded += 1;
      factSent.textContent = String(framesForwarded);
    },
    onLevel: renderLevel,
    onState: (captureState, captureDetail) => {
      detail = captureDetail;
      render(mapCaptureState(captureState));
    },
  });

  function mapCaptureState(captureState: MicCaptureState): MicIslandState {
    switch (captureState) {
      case "capturing":
        if (muted) return "muted";
        return player?.getState() === "playing" ? "speaking" : "listening";
      case "idle":
        return "idle";
      case "arming":
        return "arming";
      case "stopped":
        return "stopped";
      default:
        return captureState;
    }
  }

  function emitClientEvent(event: AppendEvent): void {
    sender?.(event);
    target.dispatchEvent(new CustomEvent(clientEventName, { detail: event, bubbles: true }));
  }

  function renderLevel(level: number): void {
    const percent = Math.round(Math.min(1, Math.max(0, level)) * 100);
    if (percent === lastMeterPercent) return;
    lastMeterPercent = percent;
    // A CSS custom property the stylesheet turns into a transform. The value
    // is data, not decoration: no rAF loop drives it, it is written when a
    // batch of samples arrives, and the reduced-motion kill switch in
    // global.css removes the transition so it snaps instead of sliding.
    meterFill.style.setProperty("--mic-level", String(percent / 100));
    meter.setAttribute("aria-valuenow", String(percent));
    meterValue.textContent = `${percent}%`;
  }

  function render(next: MicIslandState): void {
    state = next;
    const copy = STATE_COPY[next];
    root.dataset.state = next;
    chipLabel.textContent = copy.label;
    const armed = next === "listening" || next === "speaking" || next === "muted" || next === "arming";
    startButton.textContent = armed ? "Stop mic & playback" : "Start mic & playback";

    // The mute button only ever does something while the device is actually
    // held (listening/speaking/muted) — arming is mid-flight and everything
    // else has no live track to gate. `disabled` here is belt-and-braces:
    // `toggleMute` itself is a no-op unless captureState is "capturing".
    muteButton.disabled = !(next === "listening" || next === "speaking" || next === "muted");
    muteButton.textContent = muted ? "Unmute mic" : "Mute mic";
    muteButton.setAttribute("aria-pressed", String(muted));

    let text = copy.message;
    if ((next === "failed" || next === "denied" || next === "no-device") && detail.errorName) {
      text = `${copy.message} (${detail.errorName}${detail.message ? `: ${detail.message}` : ""})`;
    } else if (next === "unsupported" && detail.message) {
      text = detail.message;
    }
    message.textContent = text;

    const contextRate = capture.getContextSampleRate();
    factIn.textContent = `${capture.getInputSampleRate()} Hz PCM16 mono${
      contextRate && contextRate !== capture.getInputSampleRate()
        ? ` (resampled from ${contextRate} Hz)`
        : ""
    }`;
    // Independent of the line above on purpose: reply audio arrives at the
    // fixed TTS rate whatever the session negotiated for input.
    factOut.textContent = `${TTS_OUTPUT_SAMPLE_RATE} Hz PCM16 mono`;
    factFrames.textContent = String(capture.framesSent);
    factSent.textContent = String(framesForwarded);
    playback.textContent = playbackNote;
  }

  function setPlaybackNote(note: string): void {
    playbackNote = note;
    playback.textContent = note;
  }

  function showServerError(code: string, text: string): void {
    // Deliberately its own line with its own styling, never the state chip:
    // a named server error (`tts_failed`, `response_timeout`, …) is a
    // different kind of fact from "your browser blocked the mic", and the two
    // must not be mistakable for each other.
    serverError.hidden = false;
    serverError.textContent = `server error · ${code}${text ? ` · ${text}` : ""}`;
  }

  function clearServerError(): void {
    serverError.hidden = true;
    serverError.textContent = "";
  }

  // ------------------------------------------------------------ lifecycle
  async function start(): Promise<boolean> {
    if (state === "listening" || state === "speaking" || state === "muted" || state === "arming") {
      return true;
    }
    clearServerError();
    droppedDeltas = 0;
    setPlaybackNote("Reply audio: nothing received yet.");
    // A fresh arm always starts unmuted — the same gesture path as the very
    // first start, per the acceptance criterion, never a stale mute carried
    // over from a previous session.
    muted = false;
    framesForwarded = 0;

    if (!deps.isSupported()) {
      // Resolved HERE, not at module init: only now are `isSecureContext` and
      // the live URL readable, and they are what decide whether this is an
      // origin problem or a browser-capability one.
      detail = { message: unsupportedMessage() };
      render("unsupported");
      return false;
    }

    // Created here and nowhere else: inside the gesture, so the context is
    // allowed to run and the prompt is allowed to appear.
    const created = deps.createAudioContext();
    try {
      await created.resume();
    } catch {
      // Some browsers reject resume() when the gesture is already consumed;
      // the capture path fails loudly enough on its own if this mattered.
    }
    context = created;
    player = new DeltaPlayer(created, {
      onState: () => render(mapCaptureState(capture.getState())),
      onStop: handlePlaybackStop,
      onProgress: ({ queuedMs, receivedMs, chunks }) => {
        if (chunks === 0) return;
        setPlaybackNote(
          `Reply audio: ${chunks} chunk${chunks === 1 ? "" : "s"}, ${seconds(receivedMs)} received, ${seconds(queuedMs)} still queued.`,
        );
      },
    });

    const ok = await capture.start(created);
    if (!ok) {
      await teardownAudio();
      return false;
    }
    return true;
  }

  // AUTOMATIC-MUTE-FORBIDDEN-ZONE-START (d1: this function runs only in
  // reaction to the player finishing, being interrupted, or being torn down
  // — never from a human's own click. Deviation d1 narrows the no-mic-mute
  // rule to forbid exactly this: a mute mechanism triggered by a playback or
  // response event. no-mic-mute.test.ts scans everything between this
  // marker and its matching END for every mute mechanism, with no exception
  // for the ones a user-triggered mute is now allowed to use elsewhere.)
  function handlePlaybackStop(info: PlaybackStopInfo): void {
    if (info.reason === "interrupted") {
      setPlaybackNote(
        `Reply interrupted after ${seconds(info.playedMs)} — ${seconds(
          info.discardedMs,
        )} of already-received audio dropped. The mic never closed.`,
      );
    } else if (info.reason === "drained") {
      setPlaybackNote("Reply audio: finished.");
    } else if (info.reason === "disconnected") {
      setPlaybackNote("Reply audio: cleared, the connection went away.");
    } else {
      setPlaybackNote("Reply audio: cleared.");
    }
    render(mapCaptureState(capture.getState()));
  }
  // AUTOMATIC-MUTE-FORBIDDEN-ZONE-END

  async function teardownAudio(): Promise<void> {
    capture.stop();
    player?.stop("stopped");
    player = null;
    const closing = context;
    context = null;
    if (closing) {
      try {
        await closing.close();
      } catch {
        // A context the browser already closed throws here; teardown is
        // best-effort by nature.
      }
    }
    renderLevel(0);
    // Mic-off (a full teardown) always clears mute too: there is no device
    // left to hold a mute state on, and the next arm is a fresh start.
    muted = false;
    framesForwarded = 0;
  }

  async function stop(): Promise<void> {
    await teardownAudio();
    render(mapCaptureState(capture.getState()));
  }

  // ------------------------------------------------------------ mute (d1)
  //
  // Deliberately NOT inside a AUTOMATIC-MUTE-FORBIDDEN-ZONE: this is the one
  // place in the file a mute mechanism is allowed to live, because it only
  // ever runs from `onMuteClick` below, a human's own click. See the module
  // doc's "muted — a deliberate, user-triggered exception" section.
  function setMuted(next: boolean): void {
    if (muted === next || !capture.isRunning()) return;
    muted = next;
    emitMuteEvent(next);
    render(mapCaptureState(capture.getState()));
  }

  /**
   * Tell the event stream, honestly. `client.mic_muted`/`client.mic_unmuted`
   * are NOT `/v1/realtime` wire events — nothing server-side ever sends
   * them, and `origin: "client"` says so on the event itself — but the
   * acceptance bar for this task is that an operator can tell "muted" apart
   * from silence (which emits nothing at all, by `event-log.ts`'s own design
   * — see its module doc) and from "disconnected" (its own family, driven by
   * `lobes:connection-state`). Dispatched straight on `window`, the exact
   * seam `EventStream.astro`'s own mount script already listens on — see
   * `src/pages/index.astro`'s bridge `<script>` for the contract this
   * mirrors. No import of any t12 module: the seam is the whole point.
   */
  function emitMuteEvent(nextMuted: boolean): void {
    if (typeof window === "undefined") return;
    window.dispatchEvent(
      new CustomEvent("lobes:realtime-event", {
        detail: {
          type: nextMuted ? "client.mic_muted" : "client.mic_unmuted",
          origin: "client",
          timestamp_ms: Date.now(),
        },
      }),
    );
  }

  const onMuteClick = () => setMuted(!muted);
  muteButton.addEventListener("click", onMuteClick);

  // -------------------------------------------------------- server events
  // AUTOMATIC-MUTE-FORBIDDEN-ZONE-START (d1: every case below runs only
  // because the SERVER sent something — a response stage, a boundary, an
  // error, a close. No mute mechanism may appear anywhere in this function;
  // see the module doc's d1 section and the marker note on
  // `handlePlaybackStop` above.)
  function handleServerEvent(raw: unknown): void {
    const event = parseServerEvent(raw);
    const type = typeof event?.type === "string" ? event.type : null;
    if (!event || !type) return;

    switch (type) {
      case "session.created": {
        const rate = readConfiguredRate(event);
        // The server's echo is the authority, not what we asked for.
        if (rate !== null) capture.setInputSampleRate(rate);
        clearServerError();
        render(mapCaptureState(capture.getState()));
        break;
      }
      case "response.created": {
        clearServerError();
        setPlaybackNote("Reply audio: reply started, waiting for the first chunk.");
        break;
      }
      case "response.audio.delta": {
        const payload = typeof event.delta === "string" ? event.delta : "";
        if (!player) {
          droppedDeltas += 1;
          setPlaybackNote(
            `Reply audio: ${droppedDeltas} chunk${
              droppedDeltas === 1 ? "" : "s"
            } arrived before playback was armed and were dropped. Press start.`,
          );
          break;
        }
        player.enqueueDelta(payload);
        break;
      }
      case "response.interrupted": {
        // The client half of barge-in. The server truncates what it has not
        // sent; this drops what it already did.
        player?.stop("interrupted");
        break;
      }
      case "response.done": {
        if (player && player.getState() === "playing") {
          setPlaybackNote(
            `Reply audio: complete, ${seconds(player.queuedMs)} left to play out.`,
          );
        }
        break;
      }
      case "session.closed": {
        notifyDisconnected(typeof event.reason === "string" ? event.reason : undefined);
        break;
      }
      case "error": {
        const code = typeof event.code === "string" ? event.code : "unknown";
        showServerError(code, typeof event.message === "string" ? event.message : "");
        break;
      }
      default:
        break;
    }
  }
  // AUTOMATIC-MUTE-FORBIDDEN-ZONE-END

  // AUTOMATIC-MUTE-FORBIDDEN-ZONE-START (d1: the connection going away is
  // not itself a "playback or response event", but it is exactly as
  // automatic — nothing here is a human's click — so it gets the same
  // treatment. This function only ever tears the device down; it must never
  // grow a mute call instead.)
  function notifyDisconnected(reason?: string): void {
    player?.stop("disconnected");
    void teardownAudio().then(() => {
      setPlaybackNote(
        `Reply audio: cleared — the session closed${reason ? ` (${reason})` : ""}.`,
      );
      render(mapCaptureState(capture.getState()));
    });
  }
  // AUTOMATIC-MUTE-FORBIDDEN-ZONE-END

  // --------------------------------------------------------------- wiring
  const onClick = () => {
    void (state === "listening" || state === "speaking" || state === "muted" || state === "arming"
      ? stop()
      : start());
  };
  startButton.addEventListener("click", onClick);

  const onServerEvent = (event: Event) => {
    handleServerEvent((event as CustomEvent<unknown>).detail);
  };
  target.addEventListener(serverEventName, onServerEvent);

  render("idle");

  const handle: MicIslandHandle = {
    root,
    getState: () => state,
    getInputSampleRate: () => capture.getInputSampleRate(),
    start,
    stop,
    isMuted: () => muted,
    setMuted,
    setSender: (fn) => {
      sender = fn;
    },
    handleServerEvent,
    notifyDisconnected,
    destroy: () => {
      if (destroyed) return;
      destroyed = true;
      startButton.removeEventListener("click", onClick);
      muteButton.removeEventListener("click", onMuteClick);
      target.removeEventListener(serverEventName, onServerEvent);
      void teardownAudio();
      MOUNTED.delete(root);
    },
  };

  MOUNTED.set(root, handle);

  // Announce the rate a connection layer should put in the connect URL. The
  // island knows it; the socket owner needs it; neither imports the other.
  target.dispatchEvent(
    new CustomEvent(MIC_READY_EVENT_NAME, {
      detail: { inputSampleRate: requestedRate, outputSampleRate: TTS_OUTPUT_SAMPLE_RATE },
      bubbles: true,
    }),
  );

  return handle;
}

const MOUNTED = new WeakMap<HTMLElement, MicIslandHandle>();

/**
 * Mount every `[data-mic-island]` element in *doc* and publish the first
 * handle as `window.lobesMic`, so a connection layer can call
 * `window.lobesMic.setSender(...)` / `.handleServerEvent(...)` directly
 * instead of going through DOM events.
 */
export function autoMountMicIslands(
  doc: Document,
  options: MicIslandOptions = {},
): MicIslandHandle[] {
  const handles: MicIslandHandle[] = [];
  for (const node of Array.from(doc.querySelectorAll<HTMLElement>("[data-mic-island]"))) {
    // A per-element `data-input-sample-rate` overrides the call-site option,
    // so the rate can be chosen in markup (`<MicIsland inputSampleRate={16000} />`)
    // without the mounting script knowing anything about it.
    const declared = Number(node.dataset.inputSampleRate);
    const perNode: MicIslandOptions =
      Number.isFinite(declared) && declared > 0 ? { ...options, inputSampleRate: declared } : options;
    handles.push(mountMicIsland(node, perNode));
  }
  if (handles.length > 0 && typeof window !== "undefined") {
    (window as unknown as { lobesMic?: MicIslandHandle }).lobesMic = handles[0];
  }
  return handles;
}
