/*
 * audio-test-doubles.ts — fake Web Audio, for tests only.
 *
 * NOT shipped: nothing under `src/pages` or `src/components` imports this, so
 * `astro build` never pulls it into the bundle. It lives beside the modules it
 * doubles (rather than in a `test/` directory) so the two stay in step when an
 * interface in `audio-graph.ts` changes — a drifting double is a test that
 * proves nothing.
 *
 * jsdom has no AudioContext, no AudioWorklet, and no getUserMedia, which is
 * exactly why these exist: the island's whole state machine, including both
 * named permission failures and the barge-in stop, is provable offline against
 * these thirty-line fakes.
 *
 * {@link FakeMediaTrack} is the sharp one: its `enabled` setter throws. Muting
 * a live mic track is the one thing this island must never do (it is what
 * barge-in forecloses), so the double makes an attempt a loud test failure
 * rather than something a reviewer has to notice.
 */

import type {
  AudioBufferLike,
  AudioContextLike,
  AudioNodeLike,
  BufferSourceLike,
  MediaStreamLike,
  MediaTrackLike,
  WorkletNodeLike,
  WorkletPortLike,
} from "./audio-graph";

export class FakeAudioNode implements AudioNodeLike {
  readonly connections: AudioNodeLike[] = [];
  disconnectCalls = 0;

  connect(destination: AudioNodeLike): AudioNodeLike {
    this.connections.push(destination);
    return destination;
  }

  disconnect(): void {
    this.disconnectCalls += 1;
  }
}

export class FakeMediaTrack implements MediaTrackLike {
  stopped = false;

  stop(): void {
    this.stopped = true;
  }

  /**
   * The tripwire. Assigning `false` to a live track's `enabled` property is
   * how a half-duplex client silences its microphone; this island is required
   * to have no such code path, so the double refuses the operation outright
   * instead of quietly recording it. (Spelled in prose rather than as code so
   * `no-mic-mute.test.ts`'s scan of this very file stays honest.)
   */
  set enabled(_value: boolean) {
    throw new Error("mic gating is forbidden in this island: track.enabled was assigned");
  }

  get enabled(): boolean {
    return true;
  }
}

export class FakeMediaStream implements MediaStreamLike {
  readonly tracks: FakeMediaTrack[] = [new FakeMediaTrack()];

  getTracks(): MediaTrackLike[] {
    return this.tracks;
  }
}

export class FakeWorkletNode extends FakeAudioNode implements WorkletNodeLike {
  portClosed = false;
  readonly port: WorkletPortLike = {
    onmessage: null,
    close: () => {
      this.portClosed = true;
    },
  };

  /** Push one batch of context-rate samples, as the audio thread would. */
  emit(block: Float32Array): void {
    this.port.onmessage?.({ data: block });
  }
}

export class FakeAudioBuffer implements AudioBufferLike {
  readonly duration: number;
  private readonly channel: Float32Array;

  constructor(
    readonly numberOfChannels: number,
    readonly length: number,
    readonly sampleRate: number,
  ) {
    this.channel = new Float32Array(length);
    this.duration = length / sampleRate;
  }

  copyToChannel(source: Float32Array): void {
    this.channel.set(source);
  }

  getChannelData(): Float32Array {
    return this.channel;
  }
}

export class FakeBufferSource extends FakeAudioNode implements BufferSourceLike {
  buffer: AudioBufferLike | null = null;
  onended: (() => void) | null = null;
  startedAt: number | null = null;
  stopped = false;

  start(when = 0): void {
    this.startedAt = when;
  }

  stop(): void {
    if (this.stopped) throw new Error("source already stopped");
    this.stopped = true;
  }

  /** Simulate natural end-of-buffer. */
  finish(): void {
    this.onended?.();
  }
}

export class FakeAudioContext implements AudioContextLike {
  state = "suspended";
  currentTime = 0;
  readonly destination = new FakeAudioNode();
  readonly addedModules: string[] = [];
  readonly buffers: FakeAudioBuffer[] = [];
  readonly sources: FakeBufferSource[] = [];
  readonly mediaSources: FakeAudioNode[] = [];
  resumeCalls = 0;
  closed = false;

  readonly audioWorklet = {
    addModule: async (url: string): Promise<void> => {
      this.addedModules.push(url);
    },
  };

  constructor(readonly sampleRate = 48000) {}

  async resume(): Promise<void> {
    this.resumeCalls += 1;
    this.state = "running";
  }

  async close(): Promise<void> {
    this.closed = true;
    this.state = "closed";
  }

  createMediaStreamSource(): AudioNodeLike {
    const node = new FakeAudioNode();
    this.mediaSources.push(node);
    return node;
  }

  createBuffer(numberOfChannels: number, length: number, sampleRate: number): AudioBufferLike {
    const buffer = new FakeAudioBuffer(numberOfChannels, length, sampleRate);
    this.buffers.push(buffer);
    return buffer;
  }

  createBufferSource(): BufferSourceLike {
    const source = new FakeBufferSource();
    this.sources.push(source);
    return source;
  }
}

/** A `DOMException`-shaped rejection, the way the browser raises them. */
export function domError(name: string, message = name): Error {
  const error = new Error(message);
  error.name = name;
  return error;
}
