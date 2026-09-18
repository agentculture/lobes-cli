# Hebrew realtime — a duplex Hebrew voice session with tool calling

An **opt-in overlay** on top of the realtime audio overlay
([`realtime-pipeline.md`](realtime-pipeline.md)): the same `GET /v1/realtime`
WebSocket session, serving Hebrew. A person speaks Hebrew, the box answers in
Hebrew, and the model can call tools that the **client** owns. The English
overlay's files are byte-identical with or without it.

If you are building a client, read the contract instead:
[`contracts/realtime-tool-calling.md`](contracts/realtime-tool-calling.md).

> **Status (the #108 rule).** Everything here was built and measured on ONE box
> — the DGX Spark GB10, 2026-09-18 — from a hand-carried deployment of this
> branch's wheel. The measurements are real and cited; the *packaged* path
> (`lobes init --audio --audio-lang he` from a released wheel) is
> **DECLARED / UNVALIDATED** until an acceptance transcript from that path
> lands under `docs/evidence/`. No Jetson has run any of it (issue #276).

## The stack

| stage | engine | notes |
|---|---|---|
| VAD / turn-taking | Silero, `server_vad` | PulseVAD was tried and false-triggered on all four non-speech clips |
| STT | `ivrit-ai/whisper-large-v3-turbo`, transformers fp16 (`Dockerfile.whisper-stt`, `listen_server_whisper.py`) | ~170–200 ms per turn |
| generate | the gateway's own `/v1/chat/completions`, `OPENAI_MODEL=multimodal` | on the Spark: `nvidia/Gemma-4-26B-A4B-NVFP4`, `gemma4` tool + reasoning parsers, 32–44 tok/s |
| TTS (shipped default) | Chatterbox **Multilingual**, `language_id="he"`, phonikud niqqud (`Dockerfile.chatterbox-ml`) | seconds per sentence; needs niqqud; samples non-deterministically, hence a runaway guard |
| TTS (measured best) | **BlueTTS**, CPU ONNX (`Dockerfile.bluetts`, `lobes.realtime.bluetts_server`) | 93–305 ms per sentence, its own G2P, no niqqud wanted — see *BlueTTS* below for why it is not the default |

Evidence: `docs/evidence/2026-09-hebrew-*.txt` (seven files) and
`docs/evidence/2026-09-hebrew-realtime-streaming-bluetts-spark.txt`.

### Three things that are easy to get wrong

- **Whisper hallucinates on non-speech** — this fine-tune emits plain text such
  as `תודה רבה` for a door slam, and its `<|nospeech|>` probability reads 0.000.
  The mean token log-probability separates cleanly (speech ≥ −0.12,
  hallucination ≤ −0.58), so the sidecar drops a transcript below
  `STT_MIN_AVG_LOGPROB` (default −0.35; `off` disables). 4/4 non-speech clips
  dropped, 22/22 speech clips kept.
- **Chatterbox Multilingual without niqqud is unusable for Hebrew** (92 % → ~9 %
  round-trip WER with phonikud), and a Latin-script word derails it.
- **An NVFP4A16 export of Gemma 4 12B damaged Hebrew** where bf16 and Google's
  QAT checkpoint did not. Check Hebrew output before promoting a quantization.

## Latency: what was built, and what it bought

Measured live, a human on a reSpeaker XVF3800, from the moment the server
decides the turn ended to the first audio byte:

| configuration | first audio |
|---|---|
| whole-reply generate + Chatterbox Multilingual | 2.6 s (one sentence), 7.3 s (longer) |
| + sentence streaming (`GENERATE_STREAM`, default on) + BlueTTS | 0.67–1.0 s |
| + hidden speculation (`VAD_EAGER_MS=250`) | **1–113 ms** |

…plus, always, the silence the VAD waits through before it believes the turn
ended (`VAD_SILENCE_MS`). Once speculation is on, that wait is essentially the
whole perceived delay:

```text
perceived delay ≈ max( VAD_SILENCE_MS , VAD_EAGER_MS + pipeline )
pipeline ≈ 600 ms  =  STT ~170  +  first clause ~300  +  TTS ~120
```

| knob | default | what it does |
|---|---|---|
| `GENERATE_STREAM` | `true` | Stream the generate call; speak sentence by sentence. `false` is the exact previous path. **On for every language, English included.** |
| `REPLY_FIRST_CLAUSE_MIN_CHARS` | `24` | How long the opening clause must be before a comma may end the first spoken piece. `5` on the Spark: first audio ~1.1 s → ~0.6 s, at the price of a short first piece. |
| `VAD_EAGER_MS` | `0` (off) | **Hidden speculation.** At this much provisional silence the bridge runs STT → generate → TTS out of sight. Speech resumes → discarded without trace. Turn confirmed → adopted, but only if the real generate request is byte-identical to the speculative one. Costs a wasted STT + generate on every mid-sentence pause. |
| `CONTINUATION_WINDOW_MS` | `0` (off) | **Continuation merge.** An onset this soon after a silence commit is the speaker carrying on: the reply stops (inside the barge-in guard window too), the half-turn leaves history, both halves are re-transcribed as one turn. One take-back per utterance. What makes a short `VAD_SILENCE_MS` affordable. |
| `CONTINUATION_TOOL_HOLD_MS` | `500` | With the merge on, a finished tool call is held this long after the commit — a call the client already has cannot be taken back. Speech is never held. A fixed cost on every tool turn. |

The Spark runs `VAD_SILENCE_MS=500`, `VAD_EAGER_MS=160`,
`CONTINUATION_WINDOW_MS=1200`: plain replies 328–370 ms after the commit
(~830–870 ms after the speaker stops), tool turns 1.16–1.64 s.

**Honest limits.** The continuation merge has been proven on a recording only —
no live session has yet produced one. A noise burst with no words in it still
counts as a barge-in. A merged turn sends two `transcription.completed` events
(the half, then the whole) and nothing on the wire marks the second as
superseding the first. English words inside Hebrew speech are misheard and hard
to follow when spoken (issue #277).

## BlueTTS — built, measured, deliberately not the default

`lobes.realtime.bluetts_server` is a drop-in for the `chatterbox` service (same
port keys, same `POST /v1/audio/synthesize` → raw PCM16 mono 24 kHz). CPU-only,
resamples its native 44.1 kHz, strips any niqqud (it runs its own RenikudPlus
G2P — leave the bridge's `PHONIKUD_MODEL_PATH` unset), and reports ready only
after a warm-up synthesis has loaded that G2P.

The engine **code** (github.com/maxmelichov/BlueTTS) is MIT and pinned by
commit; `Dockerfile.bluetts` installs its dependencies from the engine's own
`uv.lock`, because an unpinned resolve picked a `renikud-plus` that 404s at
warm-up. The **weights** repo declared **no licence** when this was written, so:

- nothing in this repo downloads the weights or names their repo as a default;
- the operator downloads them deliberately and mounts the directory at
  `BLUETTS_ONNX_DIR`;
- the Hebrew overlay's shipped voice stays Chatterbox Multilingual, and no
  compose file wires BlueTTS in.

To run it, add a service built from `Dockerfile.bluetts` with that mount and
point the bridge's `TTS_URL` at it. `BLUETTS_VOICE` (`noa`), `BLUETTS_STEPS`
(5; 2–3 buys ~20–25 %, quality not auditioned), `BLUETTS_SPEED`.

## Audio hardware notes (both cost hours)

- **Check the pipewire card profile first.** The reSpeaker XVF3800 had flipped
  to its *digital* output at 34 % volume: almost nothing reached the speaker on
  its 3.5 mm jack, its echo canceller reported `converged: false` all day, and
  the session interrupted itself (6 times in 7 turns). `pactl set-card-profile
  <card> output:analog-stereo+input:analog-stereo` plus full sink volume →
  `converged: true`, zero self-interruptions. The profile can flip when the
  device re-enumerates, and the source/sink numeric suffixes change with it.
- **Wire the speaker to the device that owns the microphone.** An echo
  canceller only cancels what it plays itself; audio sent to a monitor over
  HDMI has no reference and makes echo worse. For the same reason you cannot
  test a session by playing a recording through that speaker — the canceller
  removes it.
- **A client must drain its player before exiting.** The server delivers ahead
  of the playhead; closing the player on `response.done` cut replies
  mid-sentence. In a browser the same gap shows up as stutter unless playback
  (re)starts a jitter buffer ahead of now.

## Selecting it

```bash
lobes init --fleet --audio --audio-lang he        # dry-run; add --apply
```

materialises `docker-compose.audio-he.yml`, `env.audio-he.example` (merged
append-only into `.env`), `Dockerfile.whisper-stt`, `Dockerfile.chatterbox-ml`,
`Dockerfile.bluetts` and `listen_server_whisper.py`; `lobes fleet up` includes
the overlay when it is present. `--audio` alone, or `--audio-lang en`, is
exactly the English overlay. Language is an overlay choice — not a shape, not a
variation.

`GET /capabilities` names what is served: `stt`/`tts` carry the declared
`model`, `runtime` and — only when declared — `language`.
