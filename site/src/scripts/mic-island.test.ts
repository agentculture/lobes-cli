/*
 * mic-island.test.ts — the rendered island, under jsdom (issue #151 t11).
 *
 * Acceptance criterion 1 ("mic and playback arm only from the start control;
 * NotAllowedError/NotFoundError render their own distinct states") and the
 * client half of criterion 2 ("local playback stops on the interruption
 * event") are both rendering claims as much as behavioural ones, so they are
 * asserted here against real DOM — distinct `data-state`, distinct chip label,
 * distinct message text, and a named server error that does not wear the mic's
 * clothes.
 *
 * jsdom is a DOM, not a browser: there is no audio here, no device, and no
 * permission dialog. Every browser API is a double. What this file proves is
 * that the island *decides* correctly; that the decisions sound right in a
 * room is the live acceptance run's job.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { BrowserAudioDeps } from "./audio-graph";
import {
  FakeAudioContext,
  FakeMediaStream,
  FakeWorkletNode,
  domError,
} from "./audio-test-doubles";
import type { AppendEvent } from "./pcm-wire";
import { TTS_OUTPUT_SAMPLE_RATE, encodeAudioPayload } from "./pcm-wire";
import type { MicIslandHandle, MicIslandOptions } from "./mic-island";
import { CLIENT_EVENT_NAME, SERVER_EVENT_NAME, mountMicIsland } from "./mic-island";

interface Island {
  handle: MicIslandHandle;
  root: HTMLElement;
  target: EventTarget;
  context: FakeAudioContext;
  stream: FakeMediaStream;
  node: FakeWorkletNode;
  contexts: number;
  getUserMedia: ReturnType<typeof vi.fn>;
  serve(event: Record<string, unknown>): void;
  text(selector: string): string;
  state(): string;
}

const mounted: MicIslandHandle[] = [];

function island(
  overrides: Partial<BrowserAudioDeps> = {},
  options: MicIslandOptions = {},
): Island {
  const context = new FakeAudioContext(48000);
  const stream = new FakeMediaStream();
  const node = new FakeWorkletNode();
  const target = options.eventTarget ?? new EventTarget();
  const root = document.createElement("div");
  document.body.append(root);
  const getUserMedia = vi.fn(async () => stream);
  const box = { contexts: 0 };

  const handle = mountMicIsland(root, {
    appendFrameMs: 40,
    ...options,
    eventTarget: target,
    deps: {
      isSupported: () => true,
      getUserMedia,
      createAudioContext: () => {
        box.contexts += 1;
        return context;
      },
      createWorkletNode: () => node,
      ...overrides,
    },
  });
  mounted.push(handle);

  return {
    handle,
    root,
    target,
    context,
    stream,
    node,
    get contexts() {
      return box.contexts;
    },
    getUserMedia,
    serve: (event) => target.dispatchEvent(new CustomEvent(SERVER_EVENT_NAME, { detail: event })),
    text: (selector) => root.querySelector(selector)?.textContent ?? "",
    state: () => root.dataset.state ?? "",
  };
}

/** `ms` of reply audio at the TTS rate, base64 as the server sends it. */
function replyDelta(ms: number): string {
  const count = Math.round((TTS_OUTPUT_SAMPLE_RATE * ms) / 1000);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i += 1) samples[i] = Math.sin(i / 12) * 0.25;
  return encodeAudioPayload(samples);
}

function micBlock(length: number): Float32Array {
  const out = new Float32Array(length);
  for (let i = 0; i < length; i += 1) out[i] = Math.sin(i / 15) * 0.5;
  return out;
}

const flush = () => new Promise((resolve) => setTimeout(resolve, 0));

afterEach(() => {
  while (mounted.length > 0) mounted.pop()?.destroy();
  document.body.replaceChildren();
});

describe("gesture gating", () => {
  it("constructs no AudioContext and calls no getUserMedia at mount", () => {
    const scene = island();
    expect(scene.contexts).toBe(0);
    expect(scene.getUserMedia).not.toHaveBeenCalled();
    expect(scene.state()).toBe("idle");
    expect(scene.text(".mic-chip-label")).toBe("Not armed");
    expect(scene.text(".mic-button")).toBe("Start mic & playback");
  });

  it("arms the mic and playback together from one click", async () => {
    const scene = island();
    const button = scene.root.querySelector<HTMLButtonElement>(".mic-button");
    button?.click();
    await flush();

    expect(scene.contexts).toBe(1);
    expect(scene.context.resumeCalls).toBe(1);
    expect(scene.getUserMedia).toHaveBeenCalledTimes(1);
    expect(scene.state()).toBe("listening");
    expect(scene.text(".mic-button")).toBe("Stop mic & playback");
  });

  it("stops both halves from the same control", async () => {
    const scene = island();
    await scene.handle.start();
    await scene.handle.stop();

    expect(scene.state()).toBe("stopped");
    expect(scene.stream.tracks[0].stopped).toBe(true);
    expect(scene.context.closed).toBe(true);
  });
});

describe("failure states are visually distinct", () => {
  it("renders NotAllowedError as its own blocked state", async () => {
    const scene = island({
      getUserMedia: vi.fn(async () => {
        throw domError("NotAllowedError", "Permission denied");
      }),
    });
    await scene.handle.start();

    expect(scene.state()).toBe("denied");
    expect(scene.text(".mic-chip-label")).toBe("Mic blocked");
    expect(scene.text(".mic-message")).toContain("NotAllowedError");
    expect(scene.text(".mic-message")).toContain("Allow the mic");
  });

  it("renders NotFoundError as a different state, label, and message", async () => {
    const blocked = island({
      getUserMedia: vi.fn(async () => {
        throw domError("NotAllowedError", "Permission denied");
      }),
    });
    const missing = island({
      getUserMedia: vi.fn(async () => {
        throw domError("NotFoundError", "Requested device not found");
      }),
    });
    await blocked.handle.start();
    await missing.handle.start();

    expect(missing.state()).toBe("no-device");
    expect(missing.state()).not.toBe(blocked.state());
    expect(missing.text(".mic-chip-label")).toBe("No microphone");
    expect(missing.text(".mic-chip-label")).not.toBe(blocked.text(".mic-chip-label"));
    expect(missing.text(".mic-message")).not.toBe(blocked.text(".mic-message"));
    expect(missing.text(".mic-message")).toContain("not a permission problem");
  });

  it("renders an unsupported context without constructing anything", async () => {
    const scene = island({ isSupported: () => false });
    await scene.handle.start();

    expect(scene.state()).toBe("unsupported");
    expect(scene.contexts).toBe(0);
    expect(scene.getUserMedia).not.toHaveBeenCalled();
    expect(scene.text(".mic-message")).toContain("secure context");
  });

  it("keeps silence looking like listening, not like a failure", async () => {
    const scene = island();
    await scene.handle.start();
    scene.node.emit(new Float32Array(1920)); // pure digital silence

    expect(scene.state()).toBe("listening");
    expect(scene.text(".mic-meter-value")).toBe("0%");
    expect(scene.root.querySelector<HTMLElement>(".mic-server-error")?.hidden).toBe(true);
  });
});

describe("the outbound half", () => {
  it("dispatches every encoded frame as a client event and to a direct sender", async () => {
    const sent: AppendEvent[] = [];
    const dispatched: AppendEvent[] = [];
    const scene = island({}, { send: (event) => sent.push(event) });
    scene.target.addEventListener(CLIENT_EVENT_NAME, (event) => {
      dispatched.push((event as CustomEvent<AppendEvent>).detail);
    });

    await scene.handle.start();
    scene.node.emit(micBlock(1920));

    expect(sent).toHaveLength(1);
    expect(dispatched).toEqual(sent);
    expect(sent[0].type).toBe("input_audio_buffer.append");
    expect(scene.text(".mic-fact:nth-child(3) dd")).toBe("1");
  });

  it("shows the input level as data, not as an animation", async () => {
    const scene = island();
    await scene.handle.start();
    scene.node.emit(micBlock(1920));

    const fill = scene.root.querySelector<HTMLElement>(".mic-meter-fill");
    expect(Number(fill?.style.getPropertyValue("--mic-level"))).toBeGreaterThan(0);
    expect(scene.root.querySelector(".mic-meter")?.getAttribute("aria-valuenow")).not.toBe("0");
  });

  it("adopts the input_sample_rate the server echoed in session.created", async () => {
    const scene = island();
    await scene.handle.start();
    scene.serve({
      type: "session.created",
      config: { input_sample_rate: 16000, input_audio_format: "pcm16" },
    });

    expect(scene.handle.getInputSampleRate()).toBe(16000);
    expect(scene.text(".mic-fact:nth-child(1) dd")).toContain("16000 Hz");
  });

  it("shows the input and output rates as separate facts", async () => {
    const scene = island();
    await scene.handle.start();

    expect(scene.text(".mic-fact:nth-child(1) dd")).toContain("24000 Hz");
    expect(scene.text(".mic-fact:nth-child(1) dd")).toContain("resampled from 48000 Hz");
    // Independent of the input rate, and of the context rate: the reply
    // always arrives at the TTS rate.
    expect(scene.text(".mic-fact:nth-child(2) dd")).toBe(`${TTS_OUTPUT_SAMPLE_RATE} Hz PCM16 mono`);
  });
});

describe("the inbound half", () => {
  it("plays queued deltas and shows the reply as playing", async () => {
    const scene = island();
    await scene.handle.start();
    scene.serve({ type: "response.created", response_id: "resp_1" });
    scene.serve({ type: "response.audio.delta", delta: replyDelta(200) });
    scene.serve({ type: "response.audio.delta", delta: replyDelta(200) });

    expect(scene.state()).toBe("speaking");
    expect(scene.context.sources).toHaveLength(2);
    expect(scene.context.buffers[0].sampleRate).toBe(TTS_OUTPUT_SAMPLE_RATE);
    expect(scene.text(".mic-playback")).toContain("2 chunks");
  });

  it("stops local playback on response.interrupted while the mic stays open", async () => {
    const scene = island();
    await scene.handle.start();
    scene.serve({ type: "response.audio.delta", delta: replyDelta(1000) });
    scene.serve({ type: "response.audio.delta", delta: replyDelta(1000) });
    expect(scene.state()).toBe("speaking");

    scene.serve({ type: "response.interrupted", response_id: "resp_1", truncated: true });

    // Playback: everything scheduled is stopped and the queue is gone.
    expect(scene.context.sources.every((source) => source.stopped)).toBe(true);
    expect(scene.text(".mic-playback")).toContain("interrupted");
    expect(scene.text(".mic-playback")).toContain("dropped");
    // The mic: untouched. Still capturing, still the same live track.
    expect(scene.state()).toBe("listening");
    expect(scene.stream.tracks[0].stopped).toBe(false);
    scene.node.emit(micBlock(1920));
    expect(scene.text(".mic-fact:nth-child(3) dd")).toBe("1");
  });

  it("renders a named server error on its own line, not as the mic state", async () => {
    const scene = island();
    await scene.handle.start();
    scene.serve({ type: "error", code: "tts_failed", message: "the TTS lane returned 503" });

    const line = scene.root.querySelector<HTMLElement>(".mic-server-error");
    expect(line?.hidden).toBe(false);
    expect(line?.textContent).toContain("tts_failed");
    // The mic itself is fine, and still says so.
    expect(scene.state()).toBe("listening");
    expect(scene.text(".mic-chip-label")).toBe("Listening");
  });

  it("clears playback and releases the mic when the session closes", async () => {
    const scene = island();
    await scene.handle.start();
    scene.serve({ type: "response.audio.delta", delta: replyDelta(500) });
    scene.serve({ type: "session.closed", reason: "peer went away" });
    await flush();

    expect(scene.context.sources.every((source) => source.stopped)).toBe(true);
    expect(scene.stream.tracks[0].stopped).toBe(true);
    expect(scene.state()).toBe("stopped");
    expect(scene.text(".mic-playback")).toContain("peer went away");
  });

  it("counts deltas that arrive before playback is armed instead of playing them", () => {
    const scene = island();
    scene.serve({ type: "response.audio.delta", delta: replyDelta(100) });

    expect(scene.context.sources).toHaveLength(0);
    expect(scene.text(".mic-playback")).toContain("before playback was armed");
    expect(scene.state()).toBe("idle");
  });

  it("ignores malformed and unknown events without throwing", async () => {
    const scene = island();
    await scene.handle.start();

    expect(() => scene.serve({ nope: true })).not.toThrow();
    expect(() => scene.handle.handleServerEvent("not json at all")).not.toThrow();
    expect(() => scene.handle.handleServerEvent(null)).not.toThrow();
    expect(() =>
      scene.handle.handleServerEvent('{"type":"conversation.item.input_audio_transcription.completed"}'),
    ).not.toThrow();
    expect(scene.state()).toBe("listening");
  });
});

describe("mute (issue #151 t18, deviation d1)", () => {
  it("disables the mute control before the mic is armed", () => {
    const scene = island();
    const muteButton = scene.root.querySelector<HTMLButtonElement>(".mic-mute-button");
    expect(muteButton?.disabled).toBe(true);
    expect(muteButton?.textContent).toBe("Mute mic");
  });

  it("mutes and unmutes from its own control, independent of start/stop", async () => {
    const scene = island();
    await scene.handle.start();
    const muteButton = scene.root.querySelector<HTMLButtonElement>(".mic-mute-button");
    expect(muteButton?.disabled).toBe(false);

    muteButton?.click();
    expect(scene.state()).toBe("muted");
    expect(scene.text(".mic-chip-label")).toBe("Muted");
    expect(muteButton?.textContent).toBe("Unmute mic");
    expect(muteButton?.getAttribute("aria-pressed")).toBe("true");
    // The start/stop control reads exactly as it did while listening —
    // muting never looks like stopping.
    expect(scene.text(".mic-button")).toBe("Stop mic & playback");

    muteButton?.click();
    expect(scene.state()).toBe("listening");
    expect(muteButton?.textContent).toBe("Mute mic");
    expect(muteButton?.getAttribute("aria-pressed")).toBe("false");
  });

  it("keeps the device held while muted: no track released, capture keeps running", async () => {
    const scene = island();
    await scene.handle.start();
    scene.handle.setMuted(true);

    expect(scene.state()).toBe("muted");
    expect(scene.stream.tracks[0].stopped).toBe(false);
    // The worklet keeps emitting — mic-capture.ts never learns it is muted
    // (see its own module doc): "frames captured" still climbs.
    scene.node.emit(micBlock(1920));
    expect(scene.text(".mic-fact:nth-child(3) dd")).toBe("1");
  });

  it("withholds outbound frames while muted and resumes them on unmute", async () => {
    const sent: AppendEvent[] = [];
    const scene = island({}, { send: (event) => sent.push(event) });
    await scene.handle.start();

    scene.node.emit(micBlock(1920));
    expect(sent).toHaveLength(1);
    expect(scene.text(".mic-fact:nth-child(4) dd")).toBe("1"); // frames sent

    scene.handle.setMuted(true);
    scene.node.emit(micBlock(1920));
    expect(sent).toHaveLength(1); // nothing new left the tab
    expect(scene.text(".mic-fact:nth-child(3) dd")).toBe("2"); // captured keeps counting
    expect(scene.text(".mic-fact:nth-child(4) dd")).toBe("1"); // sent stays frozen

    scene.handle.setMuted(false);
    scene.node.emit(micBlock(1920));
    expect(sent).toHaveLength(2);
    expect(scene.text(".mic-fact:nth-child(4) dd")).toBe("2");
  });

  it("does not interrupt playback when the human mutes mid-reply (plan risk r8)", async () => {
    const scene = island();
    await scene.handle.start();
    scene.serve({ type: "response.audio.delta", delta: replyDelta(1000) });
    expect(scene.state()).toBe("speaking");

    scene.handle.setMuted(true);

    // Muting is not barge-in: nothing scheduled is stopped, and the chip
    // reads "Muted", never an interruption.
    expect(scene.context.sources.every((source) => source.stopped)).toBe(false);
    expect(scene.state()).toBe("muted");
    expect(scene.text(".mic-playback")).not.toContain("interrupted");

    // Reply audio keeps arriving and keeps playing normally while muted.
    scene.serve({ type: "response.audio.delta", delta: replyDelta(200) });
    expect(scene.context.sources).toHaveLength(2);
    expect(scene.context.sources.every((source) => source.stopped)).toBe(false);
    expect(scene.state()).toBe("muted");
  });

  it("dispatches an honest client-side marker on window, never a server wire event shape", async () => {
    const scene = island();
    await scene.handle.start();

    const seen: unknown[] = [];
    const onWindowEvent = (event: Event) => seen.push((event as CustomEvent).detail);
    window.addEventListener("lobes:realtime-event", onWindowEvent);
    try {
      scene.handle.setMuted(true);
      scene.handle.setMuted(false);
    } finally {
      window.removeEventListener("lobes:realtime-event", onWindowEvent);
    }

    expect(seen).toEqual([
      expect.objectContaining({ type: "client.mic_muted", origin: "client" }),
      expect.objectContaining({ type: "client.mic_unmuted", origin: "client" }),
    ]);
    for (const detail of seen) {
      const type = (detail as { type: string }).type;
      expect(type).toMatch(/^client\.mic_/);
      // Never one of the real /v1/realtime wire event names this same
      // window feed carries once t13 is wired up (see realtime-events.ts).
      expect(type).not.toBe("error");
      expect(type.startsWith("response.")).toBe(false);
      expect(type.startsWith("session.")).toBe(false);
    }
  });

  it("is a no-op when not armed, and never fires the window event", () => {
    const scene = island();
    const seen: unknown[] = [];
    const onWindowEvent = (event: Event) => seen.push((event as CustomEvent).detail);
    window.addEventListener("lobes:realtime-event", onWindowEvent);
    try {
      scene.handle.setMuted(true);
    } finally {
      window.removeEventListener("lobes:realtime-event", onWindowEvent);
    }
    expect(scene.handle.isMuted()).toBe(false);
    expect(seen).toEqual([]);
  });

  it("resets to unmuted on a fresh start after mic off — the same gesture path as the first start", async () => {
    const scene = island();
    await scene.handle.start();
    scene.handle.setMuted(true);
    expect(scene.state()).toBe("muted");

    await scene.handle.stop();
    expect(scene.stream.tracks[0].stopped).toBe(true); // mic off: genuinely released
    expect(scene.handle.isMuted()).toBe(false);

    await scene.handle.start();
    expect(scene.state()).toBe("listening");
    expect(scene.text(".mic-mute-button")).toBe("Mute mic");
  });

  it("mic off (stop) still fully releases the device even while muted", async () => {
    const scene = island();
    await scene.handle.start();
    scene.handle.setMuted(true);

    await scene.handle.stop();

    expect(scene.state()).toBe("stopped");
    expect(scene.stream.tracks[0].stopped).toBe(true);
    expect(scene.context.closed).toBe(true);
  });

  it("keeps muted, listening, and mic-off as three mutually distinct words and data-states", async () => {
    const scene = island();
    await scene.handle.start();
    const listeningLabel = scene.text(".mic-chip-label");
    const listeningDataState = scene.state();

    scene.handle.setMuted(true);
    const mutedLabel = scene.text(".mic-chip-label");
    const mutedDataState = scene.state();

    await scene.handle.stop();
    const stoppedLabel = scene.text(".mic-chip-label");
    const stoppedDataState = scene.state();

    // Three different words in the chip...
    expect(new Set([listeningLabel, mutedLabel, stoppedLabel]).size).toBe(3);
    // ...backing three different `data-state` values, which is what drives
    // the three distinct dot-shape rules (circle / ring / square) in
    // MicIsland.astro's stylesheet — jsdom does not compute CSS, so the
    // shape rule itself is a visual/manual check, not asserted here.
    expect(new Set([listeningDataState, mutedDataState, stoppedDataState]).size).toBe(3);
  });
});

describe("mounting", () => {
  let root: HTMLElement;

  beforeEach(() => {
    root = document.createElement("div");
    document.body.append(root);
  });

  it("is idempotent per root", () => {
    const first = mountMicIsland(root);
    const second = mountMicIsland(root);
    mounted.push(first);
    expect(second).toBe(first);
    expect(root.querySelectorAll(".mic-button")).toHaveLength(1);
  });

  it("listens on document by default", async () => {
    const context = new FakeAudioContext(48000);
    const handle = mountMicIsland(root, {
      deps: {
        isSupported: () => true,
        getUserMedia: async () => new FakeMediaStream(),
        createAudioContext: () => context,
        createWorkletNode: () => new FakeWorkletNode(),
      },
    });
    mounted.push(handle);
    await handle.start();

    document.dispatchEvent(
      new CustomEvent(SERVER_EVENT_NAME, {
        detail: { type: "error", code: "response_timeout", message: "generate exceeded 30000 ms" },
      }),
    );

    expect(root.querySelector(".mic-server-error")?.textContent).toContain("response_timeout");
  });
});
