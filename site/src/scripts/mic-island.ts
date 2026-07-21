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
 * `listening` and `speaking` are both "the mic is open" — the difference is
 * whether a reply is playing over it. There is no state in which the mic is
 * open but not listening, because that state would be muting.
 */
export type MicIslandState =
  | "idle"
  | "arming"
  | "listening"
  | "speaking"
  | "stopped"
  | "denied"
  | "no-device"
  | "unsupported"
  | "failed";

interface StateCopy {
  label: string;
  message: string;
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
  stopped: {
    label: "Stopped",
    message: "The microphone is released and playback is cleared.",
  },
  denied: {
    label: "Mic blocked",
    message:
      "The browser refused microphone access. Allow the mic for this origin in the site settings, then start again.",
  },
  "no-device": {
    label: "No microphone",
    message:
      "The browser found no audio input device. Connect a microphone and start again — this is not a permission problem.",
  },
  unsupported: {
    label: "Capture unavailable",
    message:
      "This page cannot capture audio from this origin. getUserMedia needs a secure context — serve the site over localhost (the ssh -L flow) or https://.",
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
  topRow.append(chip, startButton);

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
  for (const [term, value] of [
    ["mic →", factIn],
    ["reply ←", factOut],
    ["frames sent", factFrames],
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

  const capture = new MicCapture({
    inputSampleRate: requestedRate,
    appendFrameMs: options.appendFrameMs,
    workletUrl: options.workletUrl,
    deps,
    onAppend: (event) => {
      emitClientEvent(event);
      factFrames.textContent = String(capture.framesSent);
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
    startButton.textContent =
      next === "listening" || next === "speaking" || next === "arming"
        ? "Stop mic & playback"
        : "Start mic & playback";

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
    if (state === "listening" || state === "speaking" || state === "arming") return true;
    clearServerError();
    droppedDeltas = 0;
    setPlaybackNote("Reply audio: nothing received yet.");

    if (!deps.isSupported()) {
      detail = { message: STATE_COPY.unsupported.message };
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
  }

  async function stop(): Promise<void> {
    await teardownAudio();
    render(mapCaptureState(capture.getState()));
  }

  // -------------------------------------------------------- server events
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

  function notifyDisconnected(reason?: string): void {
    player?.stop("disconnected");
    void teardownAudio().then(() => {
      setPlaybackNote(
        `Reply audio: cleared — the session closed${reason ? ` (${reason})` : ""}.`,
      );
      render(mapCaptureState(capture.getState()));
    });
  }

  // --------------------------------------------------------------- wiring
  const onClick = () => {
    void (state === "listening" || state === "speaking" || state === "arming" ? stop() : start());
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
    setSender: (fn) => {
      sender = fn;
    },
    handleServerEvent,
    notifyDisconnected,
    destroy: () => {
      if (destroyed) return;
      destroyed = true;
      startButton.removeEventListener("click", onClick);
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
