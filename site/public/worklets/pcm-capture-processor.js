/*
 * pcm-capture-processor.js — the AudioWorklet half of the mic island (#151 t11).
 *
 * Served verbatim from /public (never bundled): `audioWorklet.addModule()`
 * fetches a URL and evaluates it in the audio rendering thread's own global
 * scope, where `import`/`AudioWorkletProcessor`/`registerProcessor` mean
 * something and the main thread's module graph does not exist. Keeping it a
 * plain file under /public means what the browser evaluates is exactly what
 * is committed here — no bundler transform between the two.
 *
 * It is deliberately the dumbest thing that could work: downmix to mono,
 * batch, post. Every decision that could be wrong — resampling to the
 * negotiated wire rate, float->PCM16 conversion, base64, framing — lives in
 * `src/scripts/pcm-wire.ts` on the main thread, where it is a pure function
 * that a test can pin against a fixture. Audio-thread code is the worst
 * possible place to put logic you cannot test: it runs under a hard realtime
 * deadline, it cannot be stepped through, and a mistake there is a glitch,
 * not a stack trace.
 *
 * What it does NOT do, ever: gate, gain, mute, or otherwise suppress its own
 * input. The mic stays open through playback — that is the whole point of
 * barge-in, and the browser's `echoCancellation` constraint (set on the
 * getUserMedia call in `mic-capture.ts`) is what makes an open mic during
 * playback workable. `process()` below has exactly one branch, and it is
 * "is there an input connected at all".
 */

const DEFAULT_BATCH_SAMPLES = 1024;

class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    const requested = options && options.processorOptions && options.processorOptions.batchSamples;
    this.batchSamples =
      Number.isFinite(requested) && requested > 0 ? Math.floor(requested) : DEFAULT_BATCH_SAMPLES;
    this.buffer = new Float32Array(this.batchSamples);
    this.filled = 0;
  }

  /**
   * Batch context-rate mono frames and post them to the main thread.
   *
   * Returns `true` unconditionally so the node stays alive across silence
   * and across the gaps where a MediaStreamSource delivers no input at all
   * (returning `false` would let the browser garbage-collect the processor
   * mid-session — a mic that dies when you stop talking).
   */
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0 || !input[0]) return true;

    const channelCount = input.length;
    const frames = input[0].length;

    for (let i = 0; i < frames; i += 1) {
      // Downmix to mono here rather than trusting the `channelCount: 1`
      // getUserMedia constraint: constraints are requests, and the wire
      // contract ("mono, 1 channel" — `parse_session_config` rejects
      // anything else) is not something to leave to a hint.
      let sum = 0;
      for (let c = 0; c < channelCount; c += 1) {
        sum += input[c][i];
      }
      this.buffer[this.filled] = sum / channelCount;
      this.filled += 1;

      if (this.filled === this.batchSamples) {
        const chunk = this.buffer.slice(0);
        // Transfer, don't copy: the audio thread hands ownership of the
        // buffer over rather than paying a structured-clone per batch.
        this.port.postMessage(chunk, [chunk.buffer]);
        this.filled = 0;
      }
    }

    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
