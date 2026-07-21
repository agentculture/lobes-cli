/*
 * pcm-wire.ts — the browser half of `lobes/realtime/_wire.py`.
 *
 * The server module this mirrors is the *codec* for the /v1/realtime base64
 * event wire and nothing else: it turns one JSON event into raw PCM16 bytes
 * and raw PCM16 bytes back into JSON events. This file is the same idea on
 * the client side, and deliberately the same shape — pure functions over
 * plain values, no DOM, no AudioContext, no WebSocket. Everything here runs
 * under jsdom in `pcm-wire.test.ts` with no audio hardware anywhere, which
 * is exactly why the encode path can be fixture-proven offline (issue #151
 * t11 acceptance criterion 3).
 *
 * ── The two sample rates are NOT the same number twice ──────────────────
 *
 * They happen to be equal at the default, and that coincidence is a trap
 * (flagged by t2 during wave 1: "passing the input rate where the output
 * rate belongs misreports every duration by 1.5x and mis-sizes every
 * chunk"). They are two independent quantities:
 *
 *   INPUT  (mic -> server): NEGOTIATED per session. `parse_session_config`
 *          in `_session.py` accepts 24000 (default) or 16000, and the value
 *          travels as the `input_sample_rate` query parameter on the connect
 *          URL. The server echoes what it actually accepted back in
 *          `session.created`'s `config.input_sample_rate` — that echo, not
 *          our request, is the authority.
 *
 *   OUTPUT (server -> speaker): FIXED at 24000 by `protocol.py`'s
 *          `TTS_SAMPLE_RATE` (Chatterbox emits 24 kHz; the spec pins
 *          "no resample in the TTS-out path, 24 kHz end to end"). It is not
 *          negotiated, not derived from the input rate, and does not change
 *          when a session negotiates 16 kHz input.
 *
 * So: a 16 kHz session still plays 24 kHz reply audio. Every function below
 * that needs a rate takes it as a named argument — none of them reads a
 * module-level "the sample rate", because there is no such thing.
 */

/** Fixed by `protocol.py`'s `TTS_SAMPLE_RATE`: the rate reply audio ARRIVES at. */
export const TTS_OUTPUT_SAMPLE_RATE = 24000;

/** `protocol.py`'s `CLIENT_SAMPLE_RATE`: the default rate we SEND mic audio at. */
export const DEFAULT_INPUT_SAMPLE_RATE = 24000;

/** The only two rates `parse_session_config` accepts for `input_sample_rate`. */
export const SUPPORTED_INPUT_SAMPLE_RATES: readonly number[] = [24000, 16000];

/** PCM16: two bytes per sample (`protocol.py`'s `BYTES_PER_SAMPLE`). */
export const BYTES_PER_SAMPLE = 2;

/** `_wire.py`'s `APPEND_EVENT_TYPE`. */
export const APPEND_EVENT_TYPE = "input_audio_buffer.append";

/** `_wire.py`'s `AUDIO_DELTA_EVENT_TYPE`. */
export const AUDIO_DELTA_EVENT_TYPE = "response.audio.delta";

/** One outbound append event, exactly as `parse_append_event` expects it. */
export interface AppendEvent {
  type: typeof APPEND_EVENT_TYPE;
  audio: string;
}

/**
 * Is `rate` one the server will accept as `input_sample_rate`?
 *
 * Mirrors `_session.py`'s `_SUPPORTED_SAMPLE_RATES` check. Nothing here
 * validates the OUTPUT rate — that one is not ours to choose.
 */
export function isSupportedInputSampleRate(rate: number): boolean {
  return SUPPORTED_INPUT_SAMPLE_RATES.includes(rate);
}

/**
 * Milliseconds of audio in `byteLength` bytes of PCM16 mono at `sampleRate`.
 *
 * `sampleRate` is required and unnamed-defaulted on purpose: the whole point
 * of this helper is that a caller has to say WHICH rate it means. Pass the
 * negotiated input rate for captured audio and {@link TTS_OUTPUT_SAMPLE_RATE}
 * for reply audio.
 */
export function pcmDurationMs(byteLength: number, sampleRate: number): number {
  return (byteLength / BYTES_PER_SAMPLE / sampleRate) * 1000;
}

/** Samples in one append frame of `frameMs` at `sampleRate`. */
export function frameSamples(frameMs: number, sampleRate: number): number {
  return Math.round((sampleRate * frameMs) / 1000);
}

/**
 * Clamp and scale normalized float samples to signed 16-bit integers.
 *
 * `-1.0 -> -32768`, `+1.0 -> +32767` (the asymmetric full-scale convention:
 * the negative side of two's complement has one more code point than the
 * positive side). Values outside [-1, 1] clamp rather than wrap — a wrapped
 * sample is a loud click, and mic input does occasionally overshoot.
 */
export function floatToPcm16(samples: Float32Array): Int16Array {
  const out = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    out[i] = Math.round(s < 0 ? s * 0x8000 : s * 0x7fff);
  }
  return out;
}

/** The inverse of {@link floatToPcm16}, for decoding reply audio. */
export function pcm16ToFloat(samples: Int16Array): Float32Array {
  const out = new Float32Array(samples.length);
  for (let i = 0; i < samples.length; i += 1) {
    out[i] = samples[i] / 0x8000;
  }
  return out;
}

/**
 * Serialize PCM16 samples to **little-endian** bytes.
 *
 * Written through a `DataView` with an explicit `littleEndian` argument
 * rather than handing over `Int16Array.buffer`, because a typed array's byte
 * order is the *platform's*, not the wire's. Every realistic browser is
 * little-endian and the shortcut would work today; the wire contract
 * ("PCM16 mono little-endian" — `parse_session_config`'s own words) is
 * stated explicitly here instead of assumed.
 */
export function pcm16ToBytes(samples: Int16Array): Uint8Array {
  const bytes = new Uint8Array(samples.length * BYTES_PER_SAMPLE);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) {
    view.setInt16(i * BYTES_PER_SAMPLE, samples[i], true);
  }
  return bytes;
}

/** The inverse of {@link pcm16ToBytes}. A trailing odd byte is dropped. */
export function bytesToPcm16(bytes: Uint8Array): Int16Array {
  const count = Math.floor(bytes.length / BYTES_PER_SAMPLE);
  const out = new Int16Array(count);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  for (let i = 0; i < count; i += 1) {
    out[i] = view.getInt16(i * BYTES_PER_SAMPLE, true);
  }
  return out;
}

// `btoa` takes a binary string, and `String.fromCharCode(...huge)` blows the
// call stack somewhere around 100k arguments. Chunk it.
const B64_CHUNK = 8192;

/** Standard base64 (with padding) of raw bytes — what `b64decode(validate=True)` wants. */
export function bytesToBase64(bytes: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < bytes.length; i += B64_CHUNK) {
    const chunk = bytes.subarray(i, i + B64_CHUNK);
    binary += String.fromCharCode.apply(null, Array.from(chunk));
  }
  return btoa(binary);
}

/** The inverse of {@link bytesToBase64}. */
export function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    out[i] = binary.charCodeAt(i);
  }
  return out;
}

/**
 * The whole outbound encode path in one call: floats -> PCM16 -> LE bytes ->
 * base64. This is the function `pcm-wire.test.ts` pins against known input,
 * and the string it returns is what `parse_append_event` base64-decodes.
 */
export function encodeAudioPayload(samples: Float32Array): string {
  return bytesToBase64(pcm16ToBytes(floatToPcm16(samples)));
}

/** Wrap an encoded payload in the append event `_wire.py` parses. */
export function buildAppendEvent(samples: Float32Array | string): AppendEvent {
  return {
    type: APPEND_EVENT_TYPE,
    audio: typeof samples === "string" ? samples : encodeAudioPayload(samples),
  };
}

/**
 * Decode one `response.audio.delta` payload to float samples ready for an
 * AudioBuffer. The caller supplies {@link TTS_OUTPUT_SAMPLE_RATE} when it
 * builds that buffer — this function never guesses a rate, because bytes
 * carry none.
 */
export function decodeAudioDelta(b64: string): Float32Array {
  return pcm16ToFloat(bytesToPcm16(base64ToBytes(b64)));
}

/**
 * A stateful linear resampler, mic-context rate in, wire rate out.
 *
 * Stateful because it has to be: the browser hands us audio in blocks
 * (128-frame render quanta batched by the worklet), and the ideal sample
 * positions do not land on block boundaries. Carrying the fractional read
 * position and the unconsumed tail across `process()` calls is what keeps
 * a long capture from drifting — a stateless per-block resampler
 * accumulates a rounding error every block and slowly desynchronizes.
 *
 * Linear interpolation, not a windowed-sinc: the downstream consumer is
 * Parakeet STT and a Silero VAD, both of which get resampled to 16 kHz
 * server-side anyway, and neither is sensitive to the imaging artifacts
 * that separate linear from sinc. Cheap and predictable beats
 * theoretically-nicer on the audio thread's doorstep.
 *
 * When the rates are equal the interpolation is an exact passthrough
 * (the step is exactly 1.0, so every fraction is 0) — proven by test,
 * not asserted here.
 */
export class LinearResampler {
  readonly inputRate: number;
  readonly outputRate: number;
  private readonly step: number;
  private pending: Float32Array = new Float32Array(0);
  private position = 0;

  constructor(inputRate: number, outputRate: number) {
    if (!(inputRate > 0) || !(outputRate > 0)) {
      throw new RangeError(`sample rates must be positive, got ${inputRate} -> ${outputRate}`);
    }
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.step = inputRate / outputRate;
  }

  /** Resample one block, carrying the boundary state into the next call. */
  process(block: Float32Array): Float32Array {
    const buf = new Float32Array(this.pending.length + block.length);
    buf.set(this.pending, 0);
    buf.set(block, this.pending.length);

    const produced: number[] = [];
    let pos = this.position;
    while (Math.floor(pos) + 1 < buf.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      produced.push(buf[i] * (1 - frac) + buf[i + 1] * frac);
      pos += this.step;
    }

    const keepFrom = Math.min(Math.floor(pos), buf.length);
    this.pending = buf.slice(keepFrom);
    this.position = pos - keepFrom;
    return Float32Array.from(produced);
  }

  /** Drop the carried tail — used when a capture stops or the rate changes. */
  reset(): void {
    this.pending = new Float32Array(0);
    this.position = 0;
  }
}

/**
 * Accumulate resampled samples and hand back whole append frames.
 *
 * The wire has no minimum frame size, but a frame per 128-sample render
 * quantum would be ~190 JSON events a second for 2.7 ms of audio each — all
 * envelope, no payload. Batching to a fixed frame keeps the base64/JSON
 * overhead near the ~33% base64 floor while staying well under the server's
 * own 32 ms VAD chunk granularity, so speech onset (and therefore barge-in)
 * is not delayed by our own buffering.
 */
export class FrameAccumulator {
  readonly frameLength: number;
  private buffer: Float32Array;
  private filled = 0;

  constructor(frameLength: number) {
    if (!Number.isInteger(frameLength) || frameLength <= 0) {
      throw new RangeError(`frameLength must be a positive integer, got ${frameLength}`);
    }
    this.frameLength = frameLength;
    this.buffer = new Float32Array(frameLength);
  }

  /** Push samples; returns zero or more complete frames. */
  push(samples: Float32Array): Float32Array[] {
    const frames: Float32Array[] = [];
    let offset = 0;
    while (offset < samples.length) {
      const take = Math.min(this.frameLength - this.filled, samples.length - offset);
      this.buffer.set(samples.subarray(offset, offset + take), this.filled);
      this.filled += take;
      offset += take;
      if (this.filled === this.frameLength) {
        frames.push(this.buffer.slice(0));
        this.filled = 0;
      }
    }
    return frames;
  }

  /**
   * Hand back the partial frame, if any, and clear.
   *
   * Called when a capture stops so the last few milliseconds of a word are
   * not silently swallowed by the buffer.
   */
  flush(): Float32Array | null {
    if (this.filled === 0) return null;
    const tail = this.buffer.slice(0, this.filled);
    this.filled = 0;
    return tail;
  }
}

/** Peak absolute amplitude of a block, in [0, 1] — the level meter's input. */
export function peakLevel(samples: Float32Array): number {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = Math.abs(samples[i]);
    if (v > peak) peak = v;
  }
  return Math.min(1, peak);
}
