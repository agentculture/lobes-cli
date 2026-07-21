/*
 * mic-capture.ts — getUserMedia + AudioWorklet -> base64 append events (#151 t11).
 *
 * The capture half of the mic island. It owns the device, the worklet node,
 * and the small state machine around them; it owns no DOM, no WebSocket, and
 * — deliberately — no AudioContext. The context is created by the island
 * inside the click handler and handed to `start()`, because the *same*
 * context serves playback: one gesture, one context, both halves armed
 * together (see `mic-island.ts`).
 *
 * Encoded frames leave through the `onAppend` callback as finished
 * `input_audio_buffer.append` events — whoever holds the socket decides what
 * to do with them.
 *
 * ── Two things this module will never do ────────────────────────────────
 *
 * 1. **Arm itself.** `start()` is only ever called from a click handler.
 *    Nothing in this file touches `navigator.mediaDevices` at import time or
 *    at mount time — a suspended-until-gesture AudioContext is a browser
 *    rule, but the permission prompt is a human one, and neither should fire
 *    because a page loaded.
 *
 * 2. **Close its own ears.** There is no gate, no gain node, and no track
 *    toggle anywhere in the capture path. Half-duplex clients mute the mic
 *    while the machine talks (see `scripts/realtime-voice-loop.py`'s own
 *    docstring); muting is precisely the thing barge-in cannot coexist with.
 *    The mic stays open through playback and the browser's
 *    `echoCancellation` constraint keeps the machine's own voice out of it.
 *    `stop()` releases the device outright — that is teardown at the user's
 *    request, a different thing from silencing a live session.
 *
 * Everything the browser supplies is injected through
 * {@link BrowserAudioDeps} so the state machine — including both named
 * permission failures — is testable under jsdom, which has no audio stack.
 */

import type {
  AudioContextLike,
  AudioNodeLike,
  BrowserAudioDeps,
  MediaStreamLike,
  WorkletNodeLike,
} from "./audio-graph";
import { browserAudioDeps } from "./audio-graph";
import type { AppendEvent } from "./pcm-wire";
import {
  DEFAULT_INPUT_SAMPLE_RATE,
  FrameAccumulator,
  LinearResampler,
  buildAppendEvent,
  frameSamples,
  isSupportedInputSampleRate,
  peakLevel,
} from "./pcm-wire";

/**
 * Append-frame size.
 *
 * 40 ms sits just above the server's own 32 ms Silero VAD chunk
 * (`protocol.py`'s `VAD_CHUNK_MS`), so our batching adds at most about one
 * VAD chunk of latency to speech-onset detection — which is what barge-in
 * responsiveness is measured against — while keeping the JSON envelope
 * overhead per frame near the 33% base64 floor. 25 frames a second is
 * nothing for a WebSocket.
 */
export const DEFAULT_APPEND_FRAME_MS = 40;

/** Worklet batch size in context-rate samples: 8 render quanta, ~21 ms at 48 kHz. */
export const DEFAULT_WORKLET_BATCH_SAMPLES = 1024;

/** Where {@link MicCapture} loads its processor from (a /public asset, not a bundle). */
export const DEFAULT_WORKLET_URL = "/worklets/pcm-capture-processor.js";

/** The `registerProcessor` name inside that file. */
export const WORKLET_PROCESSOR_NAME = "pcm-capture-processor";

/**
 * The capture state machine.
 *
 * `denied` and `no-device` are first-class states, not a generic failure
 * with a different string in it: a blocked permission and an absent device
 * need different words, a different colour, and a different next step from
 * the user, and both need to be unmistakable against a mic that is simply
 * hearing silence (`capturing`, level 0).
 */
export type MicCaptureState =
  | "idle"
  | "arming"
  | "capturing"
  | "stopped"
  | "denied"
  | "no-device"
  | "unsupported"
  | "failed";

/** Extra context for a state change — the raw error name, a human message. */
export interface MicStateDetail {
  message?: string;
  errorName?: string;
}

export interface MicCaptureOptions {
  /**
   * The rate we ENCODE outbound audio at — the value that travels as the
   * `input_sample_rate` query parameter and comes back echoed in
   * `session.created`. Not the rate reply audio arrives at; see
   * `pcm-wire.ts`'s header.
   */
  inputSampleRate?: number;
  appendFrameMs?: number;
  workletUrl?: string;
  workletBatchSamples?: number;
  onAppend(event: AppendEvent): void;
  onLevel?(level: number): void;
  onState?(state: MicCaptureState, detail: MicStateDetail): void;
  deps?: Partial<BrowserAudioDeps>;
}

/**
 * The audio constraints, and why each one.
 *
 * - `echoCancellation: true` — the load-bearing one. The spec's confirmed
 *   decision is that AEC lives at the client edge (browser here, firmware on
 *   Reachy, hardware on a mic-speaker unit) and the server does none. Without
 *   it an open mic during playback transcribes the machine talking to itself.
 * - `noiseSuppression: true` — a room's HVAC floor is what makes a VAD
 *   threshold twitchy; suppressing it makes the boundaries the event stream
 *   shows mean something.
 * - `autoGainControl: false` — AGC is a moving multiplier on the signal
 *   level. This site exists partly to tune `VAD_THRESHOLD` by watching
 *   boundary events, and you cannot tune a threshold against an input whose
 *   gain is being silently rewritten under you.
 * - `channelCount: 1` — a request, not a guarantee; the worklet downmixes
 *   regardless, because the wire contract is mono and `parse_session_config`
 *   rejects anything else.
 */
export const MIC_AUDIO_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: false,
  channelCount: 1,
};

/**
 * Map a getUserMedia rejection onto a state.
 *
 * The two the acceptance criterion names get their own states. Everything
 * else lands in `failed` **carrying its real error name**, so an unexpected
 * failure stays identifiable rather than being flattened into "something
 * went wrong".
 */
export function classifyMicError(error: unknown): { state: MicCaptureState; name: string } {
  const name =
    typeof error === "object" && error !== null && "name" in error
      ? String((error as { name: unknown }).name)
      : "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
    return { state: "denied", name };
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return { state: "no-device", name };
  }
  return { state: "failed", name: name || "Error" };
}

function errorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null && "message" in error) {
    const message = String((error as { message: unknown }).message);
    if (message) return message;
  }
  return String(error);
}

/**
 * A microphone capture session: device -> worklet -> resampler -> append events.
 *
 * One instance per island; `start()`/`stop()` may be called repeatedly, each
 * `start()` acquiring the device afresh (and prompting afresh, if the browser
 * has not remembered the grant).
 */
export class MicCapture {
  private readonly deps: BrowserAudioDeps;
  private readonly settings: {
    appendFrameMs: number;
    workletUrl: string;
    workletBatchSamples: number;
  };
  private readonly onAppend: (event: AppendEvent) => void;
  private readonly onLevel: (level: number) => void;
  private readonly onStateChange: (state: MicCaptureState, detail: MicStateDetail) => void;

  private state: MicCaptureState = "idle";
  private inputSampleRate: number;
  private context: AudioContextLike | null = null;
  private stream: MediaStreamLike | null = null;
  private source: AudioNodeLike | null = null;
  private node: WorkletNodeLike | null = null;
  private resampler: LinearResampler | null = null;
  private accumulator: FrameAccumulator | null = null;
  /** Frames emitted since the last `start()` — surfaced so the UI can prove it is sending. */
  framesSent = 0;

  constructor(options: MicCaptureOptions) {
    const requestedRate = options.inputSampleRate ?? DEFAULT_INPUT_SAMPLE_RATE;
    if (!isSupportedInputSampleRate(requestedRate)) {
      throw new RangeError(
        `input_sample_rate ${requestedRate} is not one the session accepts (24000 or 16000)`,
      );
    }
    this.deps = { ...browserAudioDeps, ...(options.deps ?? {}) };
    this.settings = {
      appendFrameMs: options.appendFrameMs ?? DEFAULT_APPEND_FRAME_MS,
      workletUrl: options.workletUrl ?? DEFAULT_WORKLET_URL,
      workletBatchSamples: options.workletBatchSamples ?? DEFAULT_WORKLET_BATCH_SAMPLES,
    };
    this.inputSampleRate = requestedRate;
    this.onAppend = options.onAppend;
    this.onLevel = options.onLevel ?? (() => {});
    this.onStateChange = options.onState ?? (() => {});
  }

  getState(): MicCaptureState {
    return this.state;
  }

  /** The rate outbound frames are encoded at right now. */
  getInputSampleRate(): number {
    return this.inputSampleRate;
  }

  /** The AudioContext's own rate (whatever the hardware gave us), or null when idle. */
  getContextSampleRate(): number | null {
    return this.context?.sampleRate ?? null;
  }

  isRunning(): boolean {
    return this.state === "capturing";
  }

  /**
   * Adopt the rate the server actually accepted.
   *
   * `session.created` echoes `config.input_sample_rate`, and that echo — not
   * what we asked for in the query string — is the authority. If it differs
   * from what we are encoding at, the resampler and frame size are rebuilt
   * mid-capture rather than quietly sending audio at the wrong rate, which
   * would arrive as a slowed-down or sped-up voice and transcribe as noise.
   */
  setInputSampleRate(rate: number): void {
    if (!isSupportedInputSampleRate(rate) || rate === this.inputSampleRate) return;
    this.inputSampleRate = rate;
    if (this.context && this.state === "capturing") {
      this.flushTail();
      this.resampler = new LinearResampler(this.context.sampleRate, rate);
      this.accumulator = new FrameAccumulator(frameSamples(this.settings.appendFrameMs, rate));
    }
  }

  /**
   * Acquire the mic and start emitting append events. Call from a gesture.
   *
   * *context* is the island's already-resumed AudioContext — this module
   * neither creates nor closes it. Resolves `true` once frames are flowing,
   * `false` on any handled failure (the state and its detail carry the
   * reason; nothing is thrown at the click handler).
   */
  async start(context: AudioContextLike): Promise<boolean> {
    if (this.state === "capturing" || this.state === "arming") return this.state === "capturing";
    this.framesSent = 0;
    this.setState("arming", {});

    if (!this.deps.isSupported()) {
      this.setState("unsupported", {
        message:
          "This browser cannot capture audio here. getUserMedia and AudioWorklet need a secure context — https:// or a localhost origin.",
      });
      return false;
    }

    let stream: MediaStreamLike;
    try {
      stream = await this.deps.getUserMedia({ audio: MIC_AUDIO_CONSTRAINTS, video: false });
    } catch (error) {
      const { state, name } = classifyMicError(error);
      this.setState(state, { errorName: name, message: errorMessage(error) });
      return false;
    }

    try {
      await this.buildGraph(context, stream);
    } catch (error) {
      this.releaseStream(stream);
      this.context = null;
      this.setState("failed", {
        errorName: classifyMicError(error).name,
        message: errorMessage(error),
      });
      return false;
    }

    this.stream = stream;
    this.setState("capturing", {});
    return true;
  }

  private async buildGraph(context: AudioContextLike, stream: MediaStreamLike): Promise<void> {
    this.context = context;
    await context.audioWorklet.addModule(this.settings.workletUrl);

    const node = this.deps.createWorkletNode(context, WORKLET_PROCESSOR_NAME, {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { batchSamples: this.settings.workletBatchSamples },
    });
    node.port.onmessage = (event) => this.handleBatch(event.data);

    const source = context.createMediaStreamSource(stream);
    source.connect(node);
    // The node must be reachable from the destination for the rendering graph
    // to pull it. The processor writes nothing to its output, so what reaches
    // the speakers is silence — mic audio is never routed back out, which
    // would be the echo loop `echoCancellation` exists to fight.
    node.connect(context.destination);

    this.resampler = new LinearResampler(context.sampleRate, this.inputSampleRate);
    this.accumulator = new FrameAccumulator(
      frameSamples(this.settings.appendFrameMs, this.inputSampleRate),
    );
    this.source = source;
    this.node = node;
  }

  /** One batch of context-rate mono samples from the audio thread. */
  private handleBatch(block: Float32Array): void {
    if (!this.resampler || !this.accumulator) return;
    this.onLevel(peakLevel(block));
    for (const frame of this.accumulator.push(this.resampler.process(block))) {
      this.emit(frame);
    }
  }

  private emit(frame: Float32Array): void {
    this.framesSent += 1;
    this.onAppend(buildAppendEvent(frame));
  }

  private flushTail(): void {
    const tail = this.accumulator?.flush();
    if (tail && tail.length > 0) this.emit(tail);
    this.resampler?.reset();
  }

  /**
   * Release the device and tear the capture graph down.
   *
   * `track.stop()` ends the capture outright: the browser's recording
   * indicator goes out and a restart needs a fresh `getUserMedia`. That is
   * deliberately NOT the same operation as silencing a live track — this only
   * ever runs when the human stops the session or the connection is gone,
   * never because the machine started speaking.
   *
   * The AudioContext is left alone: the island created it and the island
   * closes it, after playback has been torn down too.
   */
  stop(): void {
    if (this.state === "capturing") this.flushTail();

    if (this.node) {
      this.node.port.onmessage = null;
      this.node.port.close?.();
      this.node.disconnect();
      this.node = null;
    }
    this.source?.disconnect();
    this.source = null;
    this.accumulator = null;
    this.resampler = null;
    this.context = null;

    if (this.stream) {
      this.releaseStream(this.stream);
      this.stream = null;
    }
    this.onLevel(0);
    if (this.state === "capturing" || this.state === "arming") this.setState("stopped", {});
  }

  private releaseStream(stream: MediaStreamLike): void {
    for (const track of stream.getTracks()) {
      track.stop();
    }
  }

  private setState(state: MicCaptureState, detail: MicStateDetail): void {
    this.state = state;
    this.onStateChange(state, detail);
  }
}
