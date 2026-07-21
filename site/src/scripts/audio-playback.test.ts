/*
 * audio-playback.test.ts — playing in order, and stopping dead (issue #151 t11).
 *
 * Covers the playback half of acceptance criterion 2: "local playback stops on
 * the interruption event". The server truncates the remainder it has not sent;
 * these tests pin the other half — that what already arrived is dropped rather
 * than played out over the human who just took the floor.
 *
 * Also pins the rate trap: reply audio plays at the fixed TTS rate, never at
 * whatever the session negotiated for input.
 */

import { describe, expect, it, vi } from "vitest";

import { DeltaPlayer } from "./audio-playback";
import type { PlaybackStopInfo } from "./audio-playback";
import { FakeAudioContext } from "./audio-test-doubles";
import { TTS_OUTPUT_SAMPLE_RATE, encodeAudioPayload } from "./pcm-wire";

/** `ms` of reply audio, encoded as the server would encode it (24 kHz). */
function delta(ms: number): string {
  const count = Math.round((TTS_OUTPUT_SAMPLE_RATE * ms) / 1000);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i += 1) samples[i] = Math.sin(i / 10) * 0.3;
  return encodeAudioPayload(samples);
}

describe("scheduling", () => {
  it("schedules deltas back to back, in arrival order", () => {
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);

    player.enqueueDelta(delta(100));
    player.enqueueDelta(delta(100));
    player.enqueueDelta(delta(100));

    expect(context.sources).toHaveLength(3);
    expect(context.sources.map((s) => s.startedAt)).toEqual([0, 0.1, 0.2]);
    expect(player.getState()).toBe("playing");
  });

  it("does not schedule in the past when deltas arrive late", () => {
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);

    player.enqueueDelta(delta(100));
    context.currentTime = 5; // the first chunk finished long ago
    player.enqueueDelta(delta(100));

    expect(context.sources[1].startedAt).toBe(5);
  });

  it("treats an empty delta as a no-op rather than a zero-length buffer", () => {
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);
    player.enqueueDelta("");
    expect(context.sources).toHaveLength(0);
    expect(player.getState()).toBe("idle");
  });

  it("returns to idle when a reply drains on its own", () => {
    const context = new FakeAudioContext();
    const stops: PlaybackStopInfo[] = [];
    const player = new DeltaPlayer(context, { onStop: (info) => stops.push(info) });

    player.enqueueDelta(delta(100));
    context.sources[0].finish();

    expect(player.getState()).toBe("idle");
    expect(stops.map((s) => s.reason)).toEqual(["drained"]);
  });
});

describe("the output rate is the TTS rate, not the negotiated input rate", () => {
  it("builds every buffer at 24000 Hz", () => {
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);
    player.enqueueDelta(delta(100));

    expect(player.outputSampleRate).toBe(TTS_OUTPUT_SAMPLE_RATE);
    expect(context.buffers[0].sampleRate).toBe(24000);
    expect(context.buffers[0].length).toBe(2400);
    expect(context.buffers[0].duration).toBeCloseTo(0.1, 9);
  });

  it("still plays 24 kHz reply audio in a session that negotiated 16 kHz input", () => {
    // The trap: a 24000-sample chunk is 1.00 s of reply audio, and 1.50 s if
    // you mistakenly measure it at the 16 kHz *input* rate. Nothing in this
    // module reads the input rate, so the second number is unreachable.
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);
    player.enqueueDelta(delta(1000));

    expect(context.buffers[0].duration).toBeCloseTo(1.0, 9);
    expect(player.queuedMs).toBeCloseTo(1000, 6);
    expect(player.queuedMs).not.toBeCloseTo(1500, 0);
  });
});

describe("stopping on barge-in", () => {
  it("stops every scheduled source immediately and drops the queue", () => {
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);
    for (let i = 0; i < 5; i += 1) player.enqueueDelta(delta(200));
    expect(player.queuedMs).toBeCloseTo(1000, 6);

    player.stop("interrupted");

    expect(context.sources.every((s) => s.stopped)).toBe(true);
    expect(player.getState()).toBe("idle");
    expect(player.queuedMs).toBe(0);
  });

  it("reports how much already-received audio the interruption discarded", () => {
    const context = new FakeAudioContext();
    const stops: PlaybackStopInfo[] = [];
    const player = new DeltaPlayer(context, { onStop: (info) => stops.push(info) });

    for (let i = 0; i < 4; i += 1) player.enqueueDelta(delta(500)); // 2 s queued
    context.currentTime = 0.75; // three quarters of a second has played

    const info = player.stop("interrupted");

    expect(info?.reason).toBe("interrupted");
    expect(info?.playedMs).toBeCloseTo(750, 3);
    expect(info?.discardedMs).toBeCloseTo(1250, 3);
    expect(stops).toHaveLength(1);
  });

  it("does not fire the drained callback for sources it stopped itself", () => {
    const context = new FakeAudioContext();
    const stops: PlaybackStopInfo[] = [];
    const player = new DeltaPlayer(context, { onStop: (info) => stops.push(info) });

    player.enqueueDelta(delta(300));
    player.stop("interrupted");

    expect(stops.map((s) => s.reason)).toEqual(["interrupted"]);
  });

  it("tolerates a source the browser already ended", () => {
    const context = new FakeAudioContext();
    const player = new DeltaPlayer(context);
    player.enqueueDelta(delta(100));
    context.sources[0].stop(); // the double throws on a second stop()

    expect(() => player.stop("interrupted")).not.toThrow();
    expect(player.getState()).toBe("idle");
  });

  it("is idempotent once idle", () => {
    const context = new FakeAudioContext();
    const onStop = vi.fn();
    const player = new DeltaPlayer(context, { onStop });

    expect(player.stop("stopped")).toBeNull();
    expect(onStop).not.toHaveBeenCalled();
  });

  it("distinguishes an interruption from a disconnect and a plain stop", () => {
    const context = new FakeAudioContext();
    const stops: PlaybackStopInfo[] = [];
    const player = new DeltaPlayer(context, { onStop: (info) => stops.push(info) });

    player.enqueueDelta(delta(100));
    player.stop("disconnected");
    player.enqueueDelta(delta(100));
    player.stop("stopped");

    expect(stops.map((s) => s.reason)).toEqual(["disconnected", "stopped"]);
  });
});

describe("progress reporting", () => {
  it("counts chunks and received milliseconds for the readout", () => {
    const context = new FakeAudioContext();
    const progress: Array<{ queuedMs: number; receivedMs: number; chunks: number }> = [];
    const player = new DeltaPlayer(context, { onProgress: (info) => progress.push(info) });

    player.enqueueDelta(delta(100));
    player.enqueueDelta(delta(100));

    expect(progress[progress.length - 1].chunks).toBe(2);
    expect(progress[progress.length - 1].receivedMs).toBeCloseTo(200, 6);
  });
});
