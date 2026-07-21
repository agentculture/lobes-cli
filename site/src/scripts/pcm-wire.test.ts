/*
 * pcm-wire.test.ts — the fixture-proven encode path (issue #151 t11, criterion 3).
 *
 * "The worklet emits well-formed append events (fixture-tested encode path)"
 * is provable without hardware precisely because the encode path is pure: a
 * known block of float samples has exactly one correct base64 string, and that
 * string is what `lobes/realtime/_wire.py`'s `parse_append_event` decodes back
 * to the bytes we meant. The fixture below was computed independently with
 * Python's own `struct.pack("<h") + base64.b64encode`, not by running this
 * code and writing down whatever came out.
 */

import { describe, expect, it } from "vitest";

import {
  APPEND_EVENT_TYPE,
  DEFAULT_INPUT_SAMPLE_RATE,
  FrameAccumulator,
  LinearResampler,
  TTS_OUTPUT_SAMPLE_RATE,
  base64ToBytes,
  buildAppendEvent,
  bytesToBase64,
  bytesToPcm16,
  decodeAudioDelta,
  encodeAudioPayload,
  floatToPcm16,
  frameSamples,
  isSupportedInputSampleRate,
  pcm16ToBytes,
  pcmDurationMs,
  peakLevel,
} from "./pcm-wire";

/**
 * Full scale, both poles, both half-scales, and silence.
 *
 *   floats  [ 0,       1.0,     -1.0,     0.5,     -0.5   ]
 *   int16   [ 0,     32767,   -32768,   16384,   -16384   ]
 *   LE bytes 00 00   ff 7f     00 80    00 40     00 c0
 *   base64  "AAD/fwCAAEAAwA=="
 *
 * Cross-checked against CPython:
 *   >>> base64.b64encode(struct.pack("<5h", 0, 32767, -32768, 16384, -16384))
 *   b'AAD/fwCAAEAAwA=='
 */
const FIXTURE_SAMPLES = Float32Array.from([0, 1.0, -1.0, 0.5, -0.5]);
const FIXTURE_BYTES_HEX = "0000ff7f0080004000c0";
const FIXTURE_BASE64 = "AAD/fwCAAEAAwA==";

function hex(bytes: Uint8Array): string {
  return Array.from(bytes)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

describe("the outbound encode path", () => {
  it("encodes a known float block to the exact base64 the server decodes", () => {
    expect(encodeAudioPayload(FIXTURE_SAMPLES)).toBe(FIXTURE_BASE64);
  });

  it("writes little-endian PCM16 bytes, full scale at both poles", () => {
    expect(hex(pcm16ToBytes(floatToPcm16(FIXTURE_SAMPLES)))).toBe(FIXTURE_BYTES_HEX);
    expect(Array.from(floatToPcm16(FIXTURE_SAMPLES))).toEqual([0, 32767, -32768, 16384, -16384]);
  });

  it("clamps out-of-range samples instead of wrapping them", () => {
    // A wrapped overshoot is a full-scale click in the opposite direction —
    // the loudest possible way to be wrong.
    expect(Array.from(floatToPcm16(Float32Array.from([1.8, -1.8])))).toEqual([32767, -32768]);
  });

  it("builds the append event shape parse_append_event expects", () => {
    const event = buildAppendEvent(FIXTURE_SAMPLES);
    expect(event).toEqual({ type: APPEND_EVENT_TYPE, audio: FIXTURE_BASE64 });
    expect(event.type).toBe("input_audio_buffer.append");
  });

  it("treats an empty block as an empty (valid) payload, not an error", () => {
    // `parse_append_event`'s own contract: "An empty string is valid (decodes
    // to b\"\") — zero bytes of audio is not malformed, just empty."
    expect(encodeAudioPayload(new Float32Array(0))).toBe("");
  });

  it("survives a block far larger than one String.fromCharCode call", () => {
    const big = new Float32Array(200_000);
    for (let i = 0; i < big.length; i += 1) big[i] = Math.sin(i / 40);
    const encoded = encodeAudioPayload(big);
    expect(base64ToBytes(encoded).length).toBe(big.length * 2);
  });
});

describe("the inbound decode path", () => {
  it("decodes a delta to the exact PCM16 sample values that were sent", () => {
    expect(Array.from(bytesToPcm16(base64ToBytes(FIXTURE_BASE64)))).toEqual([
      0, 32767, -32768, 16384, -16384,
    ]);
  });

  it("normalizes samples by 32768, so every value is exactly representable", () => {
    expect(Array.from(decodeAudioDelta(FIXTURE_BASE64))).toEqual([
      0, 32767 / 32768, -1, 0.5, -0.5,
    ]);
  });

  it("loses exactly one LSB at positive full scale on a float round-trip", () => {
    // Not a bug, and worth pinning rather than discovering later: PCM16 is
    // asymmetric (-32768..+32767). Decoding divides by 32768 and encoding
    // multiplies the positive side by 32767, so +32767 comes back as +32766.
    // Every other value, including negative full scale, is exact.
    const rebuilt = Array.from(floatToPcm16(decodeAudioDelta(FIXTURE_BASE64)));
    expect(rebuilt).toEqual([0, 32766, -32768, 16384, -16384]);
  });

  it("reads little-endian, so 0x0100 is 1 and not 256", () => {
    expect(Array.from(bytesToPcm16(Uint8Array.from([0x01, 0x00])))).toEqual([1]);
    expect(bytesToBase64(Uint8Array.from([0x01, 0x00]))).toBe("AQA=");
  });

  it("drops a trailing odd byte rather than inventing a sample", () => {
    expect(bytesToPcm16(Uint8Array.from([0x01, 0x00, 0x7f])).length).toBe(1);
  });
});

describe("the input rate and the output rate are different numbers", () => {
  it("accepts only the two rates parse_session_config accepts for input", () => {
    expect(isSupportedInputSampleRate(24000)).toBe(true);
    expect(isSupportedInputSampleRate(16000)).toBe(true);
    expect(isSupportedInputSampleRate(48000)).toBe(false);
    expect(isSupportedInputSampleRate(44100)).toBe(false);
  });

  it("measures one second of audio correctly at each rate", () => {
    // The same byte count is a different duration at a different rate; this is
    // the 1.5x error the wave-1 note warned about, pinned as arithmetic.
    expect(pcmDurationMs(48000, 24000)).toBe(1000);
    expect(pcmDurationMs(48000, 16000)).toBe(1500);
  });

  it("keeps the fixed TTS output rate independent of the negotiated input rate", () => {
    // A 16 kHz *input* session still receives 24 kHz *output*: the constants
    // are separate and the output one is not derived from anything negotiable.
    expect(TTS_OUTPUT_SAMPLE_RATE).toBe(24000);
    expect(DEFAULT_INPUT_SAMPLE_RATE).toBe(24000);
    const oneSecondOfReply = TTS_OUTPUT_SAMPLE_RATE * 2;
    expect(pcmDurationMs(oneSecondOfReply, TTS_OUTPUT_SAMPLE_RATE)).toBe(1000);
    expect(pcmDurationMs(oneSecondOfReply, 16000)).toBe(1500); // the wrong answer, if you use the input rate
  });

  it("sizes an append frame from the rate it is told", () => {
    expect(frameSamples(40, 24000)).toBe(960);
    expect(frameSamples(40, 16000)).toBe(640);
  });
});

describe("LinearResampler", () => {
  it("is an exact passthrough when the rates match", () => {
    const resampler = new LinearResampler(24000, 24000);
    const block = Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5]);
    const out = resampler.process(block);
    // The last sample is held back: linear interpolation needs the next one.
    expect(Array.from(out)).toEqual([0.1, 0.2, 0.3, 0.4].map((v) => Math.fround(v)));
    expect(Array.from(resampler.process(Float32Array.from([0.6])))).toEqual([
      Math.fround(0.5),
    ]);
  });

  it("halves the sample count when halving the rate", () => {
    const resampler = new LinearResampler(48000, 24000);
    const block = Float32Array.from([0, 1, 2, 3, 4, 5, 6, 7]);
    expect(Array.from(resampler.process(block))).toEqual([0, 2, 4, 6]);
  });

  it("does not drift across block boundaries", () => {
    // The failure this guards: a stateless per-block resampler restarts its
    // phase every block and slowly desynchronizes from the input.
    const streamed = new LinearResampler(48000, 16000);
    const whole = new LinearResampler(48000, 16000);
    const source = new Float32Array(4801);
    for (let i = 0; i < source.length; i += 1) source[i] = i;

    const parts: number[] = [];
    for (let i = 0; i < source.length; i += 137) {
      parts.push(...Array.from(streamed.process(source.subarray(i, i + 137))));
    }
    const once = Array.from(whole.process(source));

    expect(parts.length).toBe(once.length);
    expect(parts).toEqual(once);
    // 3:1 downsample of 4801 samples: every third sample, minus the tail that
    // has no successor to interpolate toward.
    expect(parts.length).toBe(1600);
  });

  it("upsamples when the context runs slower than the wire", () => {
    const resampler = new LinearResampler(16000, 24000);
    const out = resampler.process(Float32Array.from([0, 3, 6, 9]));
    expect(Array.from(out)).toEqual([0, 2, 4, 6, 8]);
  });

  it("refuses a non-positive rate", () => {
    expect(() => new LinearResampler(0, 24000)).toThrow(RangeError);
    expect(() => new LinearResampler(48000, -1)).toThrow(RangeError);
  });
});

describe("FrameAccumulator", () => {
  it("hands back whole frames only", () => {
    const acc = new FrameAccumulator(4);
    expect(acc.push(Float32Array.from([1, 2, 3]))).toEqual([]);
    const frames = acc.push(Float32Array.from([4, 5, 6, 7, 8, 9]));
    expect(frames.map((f) => Array.from(f))).toEqual([
      [1, 2, 3, 4],
      [5, 6, 7, 8],
    ]);
    expect(Array.from(acc.flush() ?? [])).toEqual([9]);
    expect(acc.flush()).toBeNull();
  });

  it("refuses a nonsense frame length", () => {
    expect(() => new FrameAccumulator(0)).toThrow(RangeError);
    expect(() => new FrameAccumulator(1.5)).toThrow(RangeError);
  });
});

describe("peakLevel", () => {
  it("reports the loudest absolute sample", () => {
    expect(peakLevel(Float32Array.from([0, -0.7, 0.3]))).toBeCloseTo(0.7, 6);
    expect(peakLevel(new Float32Array(64))).toBe(0);
  });
});
