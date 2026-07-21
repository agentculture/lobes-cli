/*
 * mic-capture.test.ts — arming, refusing, and never muting (issue #151 t11).
 *
 * Covers acceptance criterion 1 at the capture layer ("mic and playback arm
 * only from the start control; NotAllowedError/NotFoundError render their own
 * distinct states") and the capture half of criterion 2 (no mic-mute logic).
 *
 * What it cannot cover, and does not claim to: a real device, a real
 * permission prompt, and whether echo cancellation actually works at volume.
 * The constraint is asserted here; its effect is the live run's to prove.
 */

import { beforeEach, describe, expect, it, vi } from "vitest";

import type { BrowserAudioDeps } from "./audio-graph";
import {
  FakeAudioContext,
  FakeMediaStream,
  FakeWorkletNode,
  domError,
} from "./audio-test-doubles";
import type { MicCaptureState, MicStateDetail } from "./mic-capture";
import { MIC_AUDIO_CONSTRAINTS, MicCapture, classifyMicError } from "./mic-capture";
import type { AppendEvent } from "./pcm-wire";
import { base64ToBytes } from "./pcm-wire";

interface Harness {
  capture: MicCapture;
  context: FakeAudioContext;
  stream: FakeMediaStream;
  node: FakeWorkletNode;
  appends: AppendEvent[];
  states: Array<{ state: MicCaptureState; detail: MicStateDetail }>;
  levels: number[];
  getUserMedia: ReturnType<typeof vi.fn>;
}

function harness(
  overrides: Partial<BrowserAudioDeps> = {},
  options: { inputSampleRate?: number; contextRate?: number } = {},
): Harness {
  const context = new FakeAudioContext(options.contextRate ?? 48000);
  const stream = new FakeMediaStream();
  const node = new FakeWorkletNode();
  const appends: AppendEvent[] = [];
  const states: Array<{ state: MicCaptureState; detail: MicStateDetail }> = [];
  const levels: number[] = [];
  const getUserMedia = vi.fn(async () => stream);

  const capture = new MicCapture({
    inputSampleRate: options.inputSampleRate ?? 24000,
    appendFrameMs: 40,
    onAppend: (event) => appends.push(event),
    onLevel: (level) => levels.push(level),
    onState: (state, detail) => states.push({ state, detail }),
    deps: {
      isSupported: () => true,
      getUserMedia,
      createAudioContext: () => context,
      createWorkletNode: () => node,
      ...overrides,
    },
  });

  return { capture, context, stream, node, appends, states, levels, getUserMedia };
}

/** One block of context-rate samples, as the worklet would post it. */
function block(length: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) out[i] = Math.sin(i / 20) * 0.4;
  return out;
}

describe("arming", () => {
  it("does not touch the microphone until start() is called", () => {
    const h = harness();
    expect(h.getUserMedia).not.toHaveBeenCalled();
    expect(h.capture.getState()).toBe("idle");
    expect(h.context.addedModules).toEqual([]);
  });

  it("requests echoCancellation, which is what lets the mic stay open during playback", async () => {
    const h = harness();
    await h.capture.start(h.context);

    expect(MIC_AUDIO_CONSTRAINTS.echoCancellation).toBe(true);
    expect(h.getUserMedia).toHaveBeenCalledWith({
      audio: MIC_AUDIO_CONSTRAINTS,
      video: false,
    });
    // AGC would rewrite the input level under the operator while they are
    // tuning VAD_THRESHOLD against what the event stream shows.
    expect(MIC_AUDIO_CONSTRAINTS.autoGainControl).toBe(false);
  });

  it("loads the worklet from /public and wires the graph, once", async () => {
    const h = harness();
    expect(await h.capture.start(h.context)).toBe(true);

    expect(h.context.addedModules).toEqual(["/worklets/pcm-capture-processor.js"]);
    expect(h.capture.getState()).toBe("capturing");
    expect(h.context.mediaSources).toHaveLength(1);
    // source -> worklet, worklet -> destination (so the graph pulls it).
    expect(h.context.mediaSources[0].connections).toEqual([h.node]);
    expect(h.node.connections).toEqual([h.context.destination]);
  });

  it("is a no-op when already capturing", async () => {
    const h = harness();
    await h.capture.start(h.context);
    expect(await h.capture.start(h.context)).toBe(true);
    expect(h.getUserMedia).toHaveBeenCalledTimes(1);
  });
});

describe("permission and device failures are distinct states", () => {
  it("renders NotAllowedError as `denied`, carrying the real error name", async () => {
    const h = harness({
      getUserMedia: vi.fn(async () => {
        throw domError("NotAllowedError", "Permission denied");
      }),
    });

    expect(await h.capture.start(h.context)).toBe(false);
    expect(h.capture.getState()).toBe("denied");
    const last = h.states[h.states.length - 1];
    expect(last.state).toBe("denied");
    expect(last.detail.errorName).toBe("NotAllowedError");
    expect(last.detail.message).toBe("Permission denied");
  });

  it("renders NotFoundError as `no-device`, a different state from `denied`", async () => {
    const h = harness({
      getUserMedia: vi.fn(async () => {
        throw domError("NotFoundError", "Requested device not found");
      }),
    });

    expect(await h.capture.start(h.context)).toBe(false);
    expect(h.capture.getState()).toBe("no-device");
    expect(h.capture.getState()).not.toBe("denied");
    expect(h.states[h.states.length - 1].detail.errorName).toBe("NotFoundError");
  });

  it("keeps an unexpected failure identifiable instead of flattening it", async () => {
    const h = harness({
      getUserMedia: vi.fn(async () => {
        throw domError("NotReadableError", "Device in use");
      }),
    });

    expect(await h.capture.start(h.context)).toBe(false);
    expect(h.capture.getState()).toBe("failed");
    expect(h.states[h.states.length - 1].detail.errorName).toBe("NotReadableError");
  });

  it("classifies the legacy aliases the same way as the modern names", () => {
    expect(classifyMicError(domError("PermissionDeniedError")).state).toBe("denied");
    expect(classifyMicError(domError("SecurityError")).state).toBe("denied");
    expect(classifyMicError(domError("DevicesNotFoundError")).state).toBe("no-device");
    expect(classifyMicError("not an error object").state).toBe("failed");
  });

  it("reports an unsupported context without ever prompting", async () => {
    const h = harness({ isSupported: () => false });
    expect(await h.capture.start(h.context)).toBe(false);
    expect(h.capture.getState()).toBe("unsupported");
    expect(h.getUserMedia).not.toHaveBeenCalled();
  });

  it("releases the device when the graph itself fails to build", async () => {
    const context = new FakeAudioContext(48000);
    context.audioWorklet.addModule = async () => {
      throw new Error("addModule blew up");
    };
    const h = harness({}, {});
    expect(await h.capture.start(context)).toBe(false);
    expect(h.capture.getState()).toBe("failed");
    expect(h.stream.tracks[0].stopped).toBe(true);
  });
});

describe("the outbound frame stream", () => {
  let h: Harness;

  beforeEach(async () => {
    h = harness();
    await h.capture.start(h.context);
  });

  it("emits one well-formed append event per 40 ms frame at the negotiated rate", () => {
    // 1920 context samples at 48 kHz -> 960 wire samples at 24 kHz -> one
    // 40 ms frame -> 1920 bytes of PCM16.
    h.node.emit(block(1920));

    expect(h.appends).toHaveLength(1);
    expect(h.appends[0].type).toBe("input_audio_buffer.append");
    expect(base64ToBytes(h.appends[0].audio).length).toBe(960 * 2);
    expect(h.capture.framesSent).toBe(1);
  });

  it("does not emit a partial frame mid-stream", () => {
    h.node.emit(block(480));
    expect(h.appends).toHaveLength(0);
    h.node.emit(block(1440));
    expect(h.appends).toHaveLength(1);
  });

  it("reports the input level for the meter", () => {
    h.node.emit(block(1920));
    expect(h.levels.some((level) => level > 0)).toBe(true);
    expect(Math.max(...h.levels)).toBeLessThanOrEqual(1);
  });

  it("adopts the rate the server echoed in session.created", () => {
    h.capture.setInputSampleRate(16000);
    expect(h.capture.getInputSampleRate()).toBe(16000);

    h.appends.length = 0;
    // Same 1920 context samples now resample 3:1, giving a 640-sample /
    // 1280-byte frame — a different frame size, at the server's rate, not ours.
    h.node.emit(block(1920));
    expect(h.appends).toHaveLength(1);
    expect(base64ToBytes(h.appends[0].audio).length).toBe(640 * 2);
  });

  it("ignores a rate the session would reject", () => {
    h.capture.setInputSampleRate(44100);
    expect(h.capture.getInputSampleRate()).toBe(24000);
  });

  it("flushes the trailing partial frame when capture stops", () => {
    h.node.emit(block(960)); // half a frame
    expect(h.appends).toHaveLength(0);
    h.capture.stop();
    expect(h.appends).toHaveLength(1);
    expect(base64ToBytes(h.appends[0].audio).length).toBeLessThan(960 * 2);
  });
});

describe("stopping", () => {
  it("releases the device and never touches track.enabled", async () => {
    // FakeMediaTrack's `enabled` setter throws: if anything in the capture
    // path ever mutes a live track, this test explodes rather than passing
    // quietly. Releasing the device with track.stop() is teardown, which is
    // a different operation from muting a session that is still running.
    const h = harness();
    await h.capture.start(h.context);
    h.capture.stop();

    expect(h.stream.tracks[0].stopped).toBe(true);
    expect(h.capture.getState()).toBe("stopped");
    expect(h.node.disconnectCalls).toBe(1);
    expect(h.node.port.onmessage).toBeNull();
    expect(h.node.portClosed).toBe(true);
  });

  it("leaves the AudioContext for the island to close", async () => {
    const h = harness();
    await h.capture.start(h.context);
    h.capture.stop();
    expect(h.context.closed).toBe(false);
  });

  it("stops emitting frames after teardown", async () => {
    const h = harness();
    await h.capture.start(h.context);
    h.capture.stop();
    h.appends.length = 0;
    h.node.emit(block(1920));
    expect(h.appends).toHaveLength(0);
  });
});

describe("construction", () => {
  it("refuses a rate the session would reject", () => {
    expect(
      () =>
        new MicCapture({
          inputSampleRate: 44100,
          onAppend: () => {},
        }),
    ).toThrow(RangeError);
  });
});
