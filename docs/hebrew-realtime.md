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
| generate | the gateway's own `/v1/chat/completions`, `OPENAI_MODEL=multimodal` | on the Spark: `nvidia/Gemma-4-26B-A4B-NVFP4`, `gemma4` tool + reasoning parsers, 32–44 tok/s; native 262144 window since 2026-09-19 (was 32768; `docs/evidence/2026-09-19-accept-multimodal-gemma26b-native-context-spark.txt`) |
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

### Batch transcription: formats and long clips

`POST /v1/audio/transcriptions` takes any format the image's ffmpeg decodes —
m4a/aac, mp3, ogg/opus, webm, flac, and WAV at any bit depth — not only 16-bit
PCM WAV. A 16-bit RIFF/WAVE upload (the realtime bridge's own turn audio) is
still read directly and never touches ffmpeg; anything else is transcoded to
16 kHz mono PCM16 first, and an undecodable upload is a 400 `invalid_audio`.

An upload longer than Whisper's 30 s window is **chunked**, not refused: each
cut goes at the quietest 20 ms frame in the last 5 s of its window, every
window is judged on its own (silence skip, gates and retries, below), and the
non-empty pieces are joined with a space. `STT_MAX_AUDIO_SECONDS` (default 600) caps the
total; past it the reply is a 413 `clip_too_long`. The upload byte cap
(`STT_MAX_UPLOAD_BYTES`, 16 MiB) still applies first, so ten minutes of
uncompressed 16 kHz WAV (~19 MB) needs it raised; a compressed voice note does
not. MEASURED on the DGX Spark, 2026-10-09, with synthetic BlueTTS speech: an
80 s m4a was transcribed in 3 windows in 3.9 s with every sentence intact at
the seams; mp3, ogg/opus, webm, 24-bit WAV and m4a single sentences each came
back verbatim. A real microphone recording is still UNVALIDATED (#108).

### Decode quality: loops, retries and silence (issue #329)

On a real 56-minute recording, the decode above failed in two ways. A two-word
phrase looped until the token cap, and a loop is high-confidence, so it passed
the gate. The gate also dropped 61 of 125 chunks whole with no second try, and
some of them were real speech. Each chunk now goes through four steps. Each
step has a knob whose off value gives back the old decode exactly.

1. **Silence skipping (`STT_VAD`).** On an upload longer than one window,
   Silero VAD reads each chunk first. A single-window request, which covers
   every realtime turn, is never skipped: the bridge's own VAD already opened
   that turn, and a one-word reply may hold very little speech. If Silero
   finds less than `STT_VAD_MIN_SPEECH_MS` of speech, Whisper never decodes
   the chunk, so a silent chunk cannot turn into `תודה רבה`. The scan stops as
   soon as it has found enough speech, so a speech chunk costs only a few VAD
   calls. The image installs the `silero-vad` wheel `--no-deps`, because its
   torchaudio dependency would replace the base image's torch. The server loads
   the bundled TorchScript model by path. If Silero fails to load, or fails on
   a call, the server logs a warning and decodes the chunk as before.
2. **Loop prevention at decode time.** `STT_NO_REPEAT_NGRAM_SIZE` sets the
   transformers n-gram block. `STT_REPETITION_PENALTY` is off by default,
   because it penalises every recurring token, Hebrew prefix letters included.
   Beam search was considered and left out. It would cost about 4× the GPU
   time on a shared box, and the per-token log-probabilities the gate reads
   would need different bookkeeping.
3. **Two gates per attempt.** The first is the confidence gate,
   `STT_MIN_AVG_LOGPROB` on `STT_LOGPROB_STAT` (`mean` by default, the
   statistic −0.35 was measured on). The second is OpenAI Whisper's zlib
   compression ratio, `STT_COMPRESSION_RATIO_MAX`, default 2.4. The ratio is
   taken one byte per character, not over UTF-8. Every Hebrew letter shares
   one UTF-8 lead byte, which alone pushes ordinary Hebrew prose to about 2.3.
   Counted per character, the same prose scores about 1.7, like English. Those
   figures come from synthetic text, not from a recording.
4. **Retry before dropping (`STT_TEMPERATURE_FALLBACK`).** A chunk that fails
   either gate is decoded again at each sampling temperature in turn. Sampling
   uses the full distribution, and the gate undoes the temperature, so it
   reads the same quantity every time. The first result that passes both
   gates is kept. If none passes, the chunk is dropped, as before. An empty or
   tags-only transcript counts as silence and is never retried.

| knob | default | off | what it does |
|---|---|---|---|
| `STT_LOGPROB_STAT` | `mean` | `mean` | Which statistic the confidence gate reads. `median` ignores a few rare tokens, such as a name, in an otherwise confident chunk. UNMEASURED: compare the two on a real run before switching. The log lines carry both. |
| `STT_NO_REPEAT_NGRAM_SIZE` | `6` | `0` | Blocks any repeated token n-gram within a chunk. |
| `STT_REPETITION_PENALTY` | `1.0` | `1.0` | transformers' repetition penalty. `1.1` is the issue's suggestion. |
| `STT_COMPRESSION_RATIO_MAX` | `2.4` | `off` | Treats a chunk whose text compresses better than this as a loop. |
| `STT_TEMPERATURE_FALLBACK` | `0.2,0.4` | `off` | Re-decode temperatures for a failing chunk. Each one is another Whisper decode. |
| `STT_VAD` | `auto` | `off` | Skips non-speech chunks of a multi-window upload with Silero before decoding. |
| `STT_VAD_THRESHOLD` | `0.5` | — | Silero speech probability per 32 ms window. |
| `STT_VAD_MIN_SPEECH_MS` | `250` | — | How much speech makes a chunk worth decoding. |

**Observability.** The stt container logs one line for each retried, dropped
or skipped chunk:

```text
stt_chunk event=retry chunk=3/18 attempt=0 temperature=0.0 reason=compression_ratio avg_logprob=-0.04 median_logprob=-0.01 compression_ratio=9.80 tokens=440 ...
stt_chunk event=dropped chunk=7/18 attempt=2 temperature=0.4 reason=low_logprob ... text='...'
stt_chunk event=skipped chunk=9/18 reason=no_speech
```

It also logs one summary line per request:
`stt_request chunks= decoded= vad_skipped= retried= recovered= dropped= decodes= seconds=`.
Clients cannot see any of this yet. A `dropped[]` field and per-segment
`avg_logprob` / `compression_ratio` wait for `response_format=verbose_json`
(#328).

**Status (#108).** Unit-tested with a fake decoder only. No live decode has
run, and none of the defaults were measured on this checkpoint. They are the
issue's and OpenAI Whisper's starting points. To validate, re-run the issue's
recording on the live lane and read the `stt_request` and `stt_chunk` lines.
Acceptance means no phrase repeated more than 3 times in a row, fewer than
10 % of non-silent chunks dropped, and still no text from silence. Also check
that `stt_vad silero loaded` appears at startup.

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
| `VAD_MIN_LEVEL_PCT` | `0` (off) | **Ignore quiet voices** (background talk, a TV, the next room). A chunk counts as speech only while the PEAK input level — held ~250 ms, like a level meter — is at least this % of full scale; below it, speech opens no turn, interrupts no reply and does not keep a turn open. Silero cannot do this: a far voice scores as high as a near one. On a reSpeaker direct speech peaks at 15–55 % and the room floor at 1–2 %, so 3–5 is a sensible start (`4` on the Spark). Peak, not RMS — speech RMS is only 5–9 %. Depends on microphone gain and AGC: tune per device. |
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
