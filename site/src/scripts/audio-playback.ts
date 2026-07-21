/*
 * audio-playback.ts — `response.audio.delta` -> speakers, and stopping dead (#151 t11).
 *
 * The playback half of the mic island: base64 PCM16 chunks in, scheduled
 * back-to-back on the shared AudioContext, in the order they arrived. It
 * owns no DOM and no socket; it is handed decoded-nothing and does the
 * decoding itself, because the decode and the scheduling have to agree about
 * one number — the sample rate — and splitting them is how that number goes
 * wrong.
 *
 * ── The rate this module plays at is NOT the rate the mic sends at ──────
 *
 * Reply audio arrives at {@link TTS_OUTPUT_SAMPLE_RATE}, fixed at 24 kHz by
 * `protocol.py`'s `TTS_SAMPLE_RATE` (Chatterbox's native output; the spec
 * pins "no resample in the TTS-out path"). A session that negotiated 16 kHz
 * *input* still receives 24 kHz *output*. Using the input rate here would
 * play every reply 1.5x too slow and misreport every duration by the same
 * factor — see `pcm-wire.ts`'s header. `outputSampleRate` is therefore a
 * constructor option with a fixed default and no code path that derives it
 * from anything the session negotiated.
 *
 * ── Stopping is the point ───────────────────────────────────────────────
 *
 * The server truncates the *undelivered* remainder of a reply when a barge-in
 * lands (the floor machine's job). Frames already on the wire are already
 * here, and the scheduler has queued them minutes-of-audio ahead of the
 * playhead — so without a client-side stop, an interruption is inaudible: the
 * machine keeps talking through the exact moment the human took the floor.
 * {@link DeltaPlayer.stop} is the other half, and it is unconditional and
 * immediate: every scheduled source is stopped, the queue is dropped, the
 * playhead resets.
 *
 * The stop is abrupt on purpose. A short fade would be gentler on the ear and
 * would need a gain node whose value goes to zero — a shape indistinguishable,
 * to a reader or a grep, from the muting this island is required not to do.
 * There is no gain node anywhere in this island. An instant stop can click;
 * being instant is the requirement.
 */

import type { AudioBufferLike, AudioContextLike, BufferSourceLike } from "./audio-graph";
import { TTS_OUTPUT_SAMPLE_RATE, decodeAudioDelta } from "./pcm-wire";

export type PlaybackState = "idle" | "playing";

/** Why playback stopped — the UI says different words for each. */
export type PlaybackStopReason = "interrupted" | "stopped" | "disconnected" | "drained";

export interface PlaybackStopInfo {
  reason: PlaybackStopReason;
  /** Milliseconds of already-received audio that were never heard. */
  discardedMs: number;
  /** Milliseconds of this reply that had already played out. */
  playedMs: number;
}

export interface DeltaPlayerOptions {
  /** Fixed at the TTS rate. Never derived from the negotiated input rate. */
  outputSampleRate?: number;
  onState?(state: PlaybackState): void;
  onStop?(info: PlaybackStopInfo): void;
  /** Called whenever the queued/received totals change, for the UI readout. */
  onProgress?(info: { queuedMs: number; receivedMs: number; chunks: number }): void;
}

/**
 * Sequential playback of `response.audio.delta` chunks on a shared context.
 *
 * Scheduling, not buffering-then-playing: each chunk is handed to the audio
 * clock at the exact time the previous chunk ends, so a reply that arrives in
 * 100 ms frames (`_wire.py`'s `DELTA_CHUNK_MS`) plays gaplessly while later
 * frames are still in flight.
 */
export class DeltaPlayer {
  readonly outputSampleRate: number;
  private readonly context: AudioContextLike;
  private readonly onStateChange: (state: PlaybackState) => void;
  private readonly onStop: (info: PlaybackStopInfo) => void;
  private readonly onProgress: (info: {
    queuedMs: number;
    receivedMs: number;
    chunks: number;
  }) => void;

  private readonly sources = new Set<BufferSourceLike>();
  private playhead = 0;
  private replyStartedAt = 0;
  private state: PlaybackState = "idle";
  private tearingDown = false;
  /** Total audio received for the current reply, in ms. */
  private receivedMs = 0;
  private chunks = 0;

  constructor(context: AudioContextLike, options: DeltaPlayerOptions = {}) {
    this.context = context;
    this.outputSampleRate = options.outputSampleRate ?? TTS_OUTPUT_SAMPLE_RATE;
    this.onStateChange = options.onState ?? (() => {});
    this.onStop = options.onStop ?? (() => {});
    this.onProgress = options.onProgress ?? (() => {});
  }

  getState(): PlaybackState {
    return this.state;
  }

  /** Milliseconds of received audio still ahead of the playhead. */
  get queuedMs(): number {
    return Math.max(0, (this.playhead - this.context.currentTime) * 1000);
  }

  /** Milliseconds of the current reply already played out. */
  get playedMs(): number {
    if (this.state === "idle") return 0;
    return Math.max(0, (this.context.currentTime - this.replyStartedAt) * 1000);
  }

  /**
   * Decode one delta payload and schedule it after everything already queued.
   *
   * An empty payload is a no-op rather than a zero-length buffer, because
   * `createBuffer(1, 0, rate)` throws in real browsers.
   */
  enqueueDelta(base64: string): void {
    const samples = decodeAudioDelta(base64);
    if (samples.length === 0) return;

    const buffer: AudioBufferLike = this.context.createBuffer(
      1,
      samples.length,
      // The TTS rate — not the session's negotiated input rate. The buffer
      // carries its own rate, so the context resamples to hardware for us.
      this.outputSampleRate,
    );
    if (buffer.copyToChannel) {
      buffer.copyToChannel(samples, 0);
    } else {
      buffer.getChannelData(0).set(samples);
    }

    const source = this.context.createBufferSource();
    source.buffer = buffer;
    source.connect(this.context.destination);
    source.onended = () => this.handleEnded(source);

    const startAt = Math.max(this.context.currentTime, this.playhead);
    if (this.state === "idle") {
      this.replyStartedAt = startAt;
      this.receivedMs = 0;
      this.chunks = 0;
      this.setState("playing");
    }
    source.start(startAt);
    this.sources.add(source);

    this.playhead = startAt + samples.length / this.outputSampleRate;
    this.receivedMs += (samples.length / this.outputSampleRate) * 1000;
    this.chunks += 1;
    this.emitProgress();
  }

  /**
   * Stop everything now and drop what has not been heard.
   *
   * Called on `response.interrupted` (the barge-in half this client owes the
   * server), on an explicit stop, and on a lost connection. Idempotent: a
   * second call while already idle reports nothing.
   */
  stop(reason: PlaybackStopReason = "stopped"): PlaybackStopInfo | null {
    if (this.state === "idle" && this.sources.size === 0) return null;

    const info: PlaybackStopInfo = {
      reason,
      discardedMs: this.queuedMs,
      playedMs: this.playedMs,
    };

    this.tearingDown = true;
    for (const source of this.sources) {
      source.onended = null;
      try {
        source.stop();
      } catch {
        // A source that already ended throws on stop(). Nothing to do — the
        // goal is silence, and it is already silent.
      }
      try {
        source.disconnect();
      } catch {
        // Same: teardown is best-effort by nature.
      }
    }
    this.sources.clear();
    this.tearingDown = false;

    this.playhead = 0;
    this.receivedMs = 0;
    this.chunks = 0;
    this.setState("idle");
    this.emitProgress();
    this.onStop(info);
    return info;
  }

  private handleEnded(source: BufferSourceLike): void {
    if (this.tearingDown) return;
    this.sources.delete(source);
    if (this.sources.size === 0 && this.state === "playing") {
      // The reply drained on its own — a completed reply, not an interruption.
      this.playhead = 0;
      this.setState("idle");
      this.onStop({ reason: "drained", discardedMs: 0, playedMs: 0 });
    }
    this.emitProgress();
  }

  private setState(state: PlaybackState): void {
    if (this.state === state) return;
    this.state = state;
    this.onStateChange(state);
  }

  private emitProgress(): void {
    this.onProgress({
      queuedMs: this.queuedMs,
      receivedMs: this.receivedMs,
      chunks: this.chunks,
    });
  }
}
