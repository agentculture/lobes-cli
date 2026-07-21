/*
 * dev-mic-harness.ts — a fake server for /dev-mic (issue #151 t11).
 *
 * The mic island's inbound half (deltas, interruptions, named errors) is
 * driven entirely by DOM events, so it can be exercised with no gateway, no
 * model, and no network at all: this harness synthesizes the server events by
 * hand and dispatches them on `document`, exactly as a real connection layer
 * would.
 *
 * It exists so the island can be *seen* working in a browser before the
 * connection task lands and before anyone books time on the Spark. What it
 * cannot do is prove the outbound half: a real microphone and a real permission
 * prompt are not synthesizable here, and neither is echo cancellation. Those
 * belong to the live acceptance run.
 *
 * Nothing in this module is imported by the real page — it is loaded only by
 * `src/pages/dev-mic.astro`.
 */

import { SERVER_EVENT_NAME, CLIENT_EVENT_NAME } from "./mic-island";
import { TTS_OUTPUT_SAMPLE_RATE, encodeAudioPayload, frameSamples } from "./pcm-wire";

/** A 100 ms delta, matching `_wire.py`'s `DELTA_CHUNK_MS`. */
const DELTA_CHUNK_MS = 100;

let responseCounter = 0;

/**
 * A quiet 440 Hz tone, `ms` long, at the TTS rate.
 *
 * At the TTS rate — {@link TTS_OUTPUT_SAMPLE_RATE} — because that is the rate
 * reply audio actually arrives at, whatever the session negotiated for input.
 * Generating this at the input rate would be the exact confusion the wire
 * module's header warns about, and it would sound like it.
 */
function tone(ms: number, hz = 440, amplitude = 0.2): Float32Array {
  const count = Math.round((TTS_OUTPUT_SAMPLE_RATE * ms) / 1000);
  const samples = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    // A short fade at each end so the fixture does not click on its own and
    // get mistaken for the (deliberately abrupt) barge-in stop.
    const fade = Math.min(1, Math.min(i, count - i) / (TTS_OUTPUT_SAMPLE_RATE * 0.01));
    samples[i] = Math.sin((2 * Math.PI * hz * i) / TTS_OUTPUT_SAMPLE_RATE) * amplitude * fade;
  }
  return samples;
}

function dispatch(event: Record<string, unknown>): void {
  document.dispatchEvent(new CustomEvent(SERVER_EVENT_NAME, { detail: event, bubbles: true }));
}

function emitReply(ms: number): void {
  responseCounter += 1;
  const responseId = `resp_dev_${responseCounter}`;
  dispatch({ type: "response.created", response_id: responseId });
  dispatch({
    type: "response.text.done",
    response_id: responseId,
    text: "A synthetic reply from the dev harness.",
  });

  const samples = tone(ms);
  const chunk = frameSamples(DELTA_CHUNK_MS, TTS_OUTPUT_SAMPLE_RATE);
  for (let offset = 0; offset < samples.length; offset += chunk) {
    dispatch({
      type: "response.audio.delta",
      response_id: responseId,
      delta: encodeAudioPayload(samples.subarray(offset, offset + chunk)),
    });
  }
  dispatch({ type: "response.done", response_id: responseId });
}

interface HarnessAction {
  label: string;
  run(): void;
}

const ACTIONS: HarnessAction[] = [
  {
    label: "session.created @ 24000",
    run: () =>
      dispatch({
        type: "session.created",
        session_id: "sess_dev",
        config: { input_sample_rate: 24000, input_audio_format: "pcm16", channels: 1 },
      }),
  },
  {
    label: "session.created @ 16000",
    run: () =>
      dispatch({
        type: "session.created",
        session_id: "sess_dev",
        config: { input_sample_rate: 16000, input_audio_format: "pcm16", channels: 1 },
      }),
  },
  { label: "reply — 4 s of audio", run: () => emitReply(4000) },
  { label: "reply — 0.6 s of audio", run: () => emitReply(600) },
  {
    label: "response.interrupted (barge-in)",
    run: () =>
      dispatch({
        type: "response.interrupted",
        response_id: `resp_dev_${responseCounter}`,
        truncated: true,
      }),
  },
  {
    label: "error — tts_failed",
    run: () =>
      dispatch({
        type: "error",
        code: "tts_failed",
        message: "the TTS lane returned 503",
      }),
  },
  {
    label: "error — response_timeout",
    run: () =>
      dispatch({ type: "error", code: "response_timeout", message: "generate exceeded 30000 ms" }),
  },
  {
    label: "session.closed (disconnect)",
    run: () => dispatch({ type: "session.closed", reason: "harness disconnect" }),
  },
];

/** Build the harness controls and the outbound-frame counter into *root*. */
export function mountDevHarness(root: HTMLElement): void {
  const doc = root.ownerDocument;
  root.replaceChildren();

  const buttons = doc.createElement("div");
  buttons.className = "harness-buttons";
  for (const action of ACTIONS) {
    const button = doc.createElement("button");
    button.type = "button";
    button.className = "harness-button";
    button.textContent = action.label;
    button.addEventListener("click", action.run);
    buttons.append(button);
  }

  const readout = doc.createElement("p");
  readout.className = "harness-readout";
  readout.textContent = "Outbound: 0 append events.";

  let frames = 0;
  let bytes = 0;
  document.addEventListener(CLIENT_EVENT_NAME, (event) => {
    const detail = (event as CustomEvent<{ audio?: string }>).detail;
    frames += 1;
    // base64 -> bytes, near enough: 4 characters carry 3 bytes.
    bytes += Math.floor(((detail?.audio?.length ?? 0) * 3) / 4);
    readout.textContent = `Outbound: ${frames} append events, ~${(bytes / 1024).toFixed(1)} KiB of PCM16.`;
  });

  root.append(buttons, readout);
}
