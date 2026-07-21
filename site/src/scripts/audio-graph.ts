/*
 * audio-graph.ts — narrow structural stand-ins for the Web Audio types.
 *
 * The capture path and the playback path share one AudioContext (one user
 * gesture arms both), so they share the shape of what a context is. These
 * interfaces exist for one reason: jsdom has no audio stack whatsoever, and
 * a test that wants to prove "playback stopped on the interruption event"
 * should be able to hand over a thirty-line fake instead of a browser.
 *
 * They are deliberately the *smallest* surface each module actually touches
 * — a real `AudioContext` satisfies all of them, and the fakes in the test
 * files are readable in full on one screen. The single place a real
 * AudioContext is constructed casts through `unknown` once, with a comment,
 * rather than widening these interfaces to match every overload TypeScript
 * knows about.
 */

export interface AudioNodeLike {
  connect(destination: AudioNodeLike): unknown;
  disconnect(): void;
}

export interface MediaTrackLike {
  stop(): void;
}

export interface MediaStreamLike {
  getTracks(): MediaTrackLike[];
}

export interface WorkletPortLike {
  onmessage: ((event: { data: Float32Array }) => void) | null;
  close?(): void;
}

export interface WorkletNodeLike extends AudioNodeLike {
  port: WorkletPortLike;
}

export interface AudioBufferLike {
  readonly duration: number;
  copyToChannel?(source: Float32Array, channelNumber: number): void;
  getChannelData(channelNumber: number): Float32Array;
}

export interface BufferSourceLike extends AudioNodeLike {
  buffer: AudioBufferLike | null;
  onended: (() => void) | null;
  start(when?: number): void;
  stop(when?: number): void;
}

/**
 * The shared context. Capture uses `sampleRate`/`audioWorklet`/
 * `createMediaStreamSource`; playback uses `currentTime`/`createBuffer`/
 * `createBufferSource`; the island owns `resume()`/`close()` because the
 * island owns the gesture.
 */
export interface AudioContextLike {
  readonly sampleRate: number;
  readonly state: string;
  readonly currentTime: number;
  readonly destination: AudioNodeLike;
  audioWorklet: { addModule(url: string): Promise<void> };
  resume(): Promise<void>;
  close(): Promise<void>;
  createMediaStreamSource(stream: MediaStreamLike): AudioNodeLike;
  createBuffer(numberOfChannels: number, length: number, sampleRate: number): AudioBufferLike;
  createBufferSource(): BufferSourceLike;
}

export interface WorkletNodeOptionsLike {
  numberOfInputs: number;
  numberOfOutputs: number;
  outputChannelCount: number[];
  processorOptions: { batchSamples: number };
}

/**
 * Everything the island needs from the browser, in one injectable bag.
 *
 * Resolved lazily by {@link browserAudioDeps} so that importing any of this
 * under jsdom — or during Astro's server-side render — neither throws nor
 * touches a device.
 */
export interface BrowserAudioDeps {
  /** True when this browser/context can capture at all (secure context, APIs present). */
  isSupported(): boolean;
  getUserMedia(constraints: MediaStreamConstraints): Promise<MediaStreamLike>;
  createAudioContext(): AudioContextLike;
  createWorkletNode(
    context: AudioContextLike,
    name: string,
    options: WorkletNodeOptionsLike,
  ): WorkletNodeLike;
}

function browserIsSupported(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getUserMedia === "function" &&
    typeof globalThis.AudioWorkletNode === "function" &&
    typeof globalThis.AudioContext === "function"
  );
}

/** The real browser implementations. Nothing here runs at import time. */
export const browserAudioDeps: BrowserAudioDeps = {
  isSupported: browserIsSupported,
  getUserMedia: (constraints) => navigator.mediaDevices.getUserMedia(constraints),
  // The one place the real Web Audio types meet the narrow structural ones
  // above. A real AudioContext satisfies every member; TypeScript will not
  // prove that through the wider DOM overloads, so it is asserted here, once,
  // instead of loosening the interfaces everywhere else.
  createAudioContext: () => new AudioContext() as unknown as AudioContextLike,
  createWorkletNode: (context, name, options) =>
    new AudioWorkletNode(
      context as unknown as BaseAudioContext,
      name,
      options,
    ) as unknown as WorkletNodeLike,
};
