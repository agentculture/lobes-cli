# Audio realtime pipeline: STT + TTS behind the fleet gateway

## Ownership

**lobes owns the live audio surface** — the OpenAI `/v1/audio/*` facade
deployed as the `realtime` container in the fleet. The realtime bridge ships in
the lobes wheel (`lobes/realtime/app.py` and friends) and is built and
managed by `lobes init --fleet --audio` / `lobes fleet up --apply`.

This consolidates what used to be a separate `realtime-api` sibling stack. STT
is Parakeet (NeMo ASR); TTS is **Chatterbox** (Resemble AI, open-weights,
Apache-2.0), running as a FastAPI sidecar container — no NGC key required.

The overlay compose file is **`lobes/templates/fleet/docker-compose.audio.yml`**.
It is layered on top of the base fleet automatically when present during `model
fleet up`.

### Local operator overrides

To change the overlay for one box — publishing a port, pinning a device, adding a
mount — put it in a **`docker-compose.override.yml`** in the deployment dir.
`lobes fleet up`/`down` append it to the `-f` chain **last**, so it wins over the
base fleet and every lobes-authored overlay. lobes never scaffolds or writes this
file; it is yours.

Use exactly that name. `docker compose` auto-discovers `docker-compose.override.yml`
only when it resolves the project itself, and any explicit `-f` — which lobes passes
as soon as the audio or shape overlay exists — suppresses that discovery. lobes names
the file explicitly to keep the convention true, but only under its conventional name;
a differently-named file (`docker-compose.mine.yml`) is invisible to lobes and its
edits will silently vanish on the next `fleet up`.

A worked example — publishing the STT container on loopback so `reachy-mini-cli`'s
default `REACHY_STT_URL=http://localhost:9002` resolves without per-container-IP wiring:

```yaml
# <deploy-dir>/docker-compose.override.yml
services:
  stt:
    ports:
      - "127.0.0.1:9002:9002"
```

## Topology

```text
┌─────────────────────────────────────────────────────────────────┐
│ Client / OpenAI SDK                                             │
│                                                                 │
│ POST http://localhost:8000/v1/audio/transcriptions              │
│  or  /v1/audio/speech                                          │
└──────────┬──────────────────────────────────────────────────────┘
           │
           │ port 8000 (host)
           │
      [model-gear-gateway]  (stdlib reverse proxy)
           │
           │ internal, compose network
           │
           ├─ route /v1/audio/* → [model-gear-realtime] :8080
           │
           └─ LLM requests → [vllm backends]
           
      [model-gear-realtime] :8080 (the facade)
           │
           ├─ POST /v1/audio/transcriptions → [model-gear-stt] :9002 (Parakeet)
           │
           └─ POST /v1/audio/speech → [model-gear-chatterbox] :9000 (Chatterbox)

Both STT and TTS share the GPU with the two LLM backends.
```

**Per-lane cross-box routing (issue #129).** The gateway routes the two audio
endpoints per-ROLE, not as one namespace: `/v1/audio/speech` is the `tts` lane
and `/v1/audio/transcriptions` the `stt` lane, and each can independently be
declared off (`STT_FEASIBLE`/`TTS_FEASIBLE=false`) and proxied to a peer box
(`*_PEER_ORIGIN` + `*_PEER_PROXY` + `*_PEER_API_KEY` — the same proxy-lobes
channels and guarantees as every other role; see
`docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in`). `AUDIO_URL`
stays exactly what the diagram shows: the LOCAL bridge lane for the lanes this
box serves itself. With no audio peer knob set anywhere (every pre-#129
deployment), routing is byte-identical to the diagram. Cross-box audio is
DECLARED/UNVALIDATED until a live acceptance transcript lands under
`docs/evidence/` (#108).

## Bring-up

### Prerequisites

- A GPU box (DGX Spark or dedicated; two ~30B models + audio barely co-fit on a
  shared GB10 — see [Memory and co-residence risk](#memory-co-residence-risk) below).
- No NGC API key required — Chatterbox is open-weights (Apache-2.0).

### Steps

```bash
# 1. Initialize the fleet with the audio overlay
lobes init --fleet --audio --apply

# 2. (optional) edit $HOME/.lobes/.env — DEFAULT_VOICE for TTS voice cloning,
#    PARAKEET_MODEL / CHATTERBOX_PORT / PARAKEET_PORT to override defaults.
#    No NGC key is required: both STT (Parakeet, NeMo) and TTS (Chatterbox,
#    Resemble AI, Apache-2.0) are open-weights — pulled from HuggingFace, not NGC.

# 3. Bring up the full audio stack (dry-run by default; --apply commits)
lobes fleet up --apply

# 4. Check status
lobes fleet status
```

Each `lobes init` and `lobes fleet` verb defaults to **dry-run**; omit `--apply`
to see what would happen, or add `--apply` to execute. This ensures safe-by-default
operation (useful when agents call CLIs in loops).

To customize the compose dir (default `$LOBES_DIR` or `$HOME/.lobes`):

```bash
lobes init --fleet --audio --compose-dir /path/to/deployment --apply
lobes fleet up --compose-dir /path/to/deployment --apply
```

## The drift this fixed

**Before (issue #39, #40):**

- The old `realtime-api` sibling stack's `:8080` exposed only `/` and `/health`
  endpoints; OpenAI REST routes like `/v1/audio/transcriptions` returned 404.
- Parakeet (`:9002`) failed nearly every transcription with `torch.AcceleratorError:
  CUDA error: unknown error` deep in NeMo's CUDA context, even though its Docker
  healthcheck reported "healthy".
- The healthcheck was liveness-only (probe `/health` without exercising the model);
  "healthy" did not mean "actually serving".

**After:**

- lobes now owns the audio surface. `lobes init --fleet --audio` scaffolds
  the complete overlay (compose file, Dockerfiles for realtime, Parakeet, and
  Chatterbox, env keys), and `lobes fleet up --apply` builds and starts all three
  services (`chatterbox`, `stt`, `realtime`) behind the gateway.
- The realtime bridge forwards `/v1/audio/transcriptions` and `/v1/audio/speech`
  to the backends (Parakeet and Chatterbox respectively) and wraps their responses
  in the OpenAI schema.
- Parakeet's healthcheck now includes a real model-readiness probe (loads the
  model, runs a trivial CUDA op); "healthy" means actually serving.

## Health and readiness

Parakeet's healthcheck (in `docker-compose.audio.yml`) is:

```yaml
healthcheck:
  test:
    - CMD
    - python3
    - -c
    - import urllib.request; urllib.request.urlopen('http://localhost:${PARAKEET_PORT:-9002}/v1/health/ready')
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 600s
```

This probe:

- Loads the Parakeet model on container startup.
- Runs a trivial CUDA operation (`/v1/health/ready` inside NeMo).
- Reports "healthy" only if the model is actually loaded and CUDA is responsive.

It is **not** a liveness check (like `curl http://localhost:9002/health` would be).
If Parakeet is "healthy", it is actively serving transcriptions.

## Runbook: stale Parakeet CUDA context (CUDA error: unknown error)

**Symptom:**

```text
Error 500: torch.AcceleratorError: CUDA error: unknown error
```

Transcription requests fail with 500s even though `docker ps` shows the STT
container as "healthy".

**Root cause (suspected):**

On the shared GB10 (DGX Spark), prolonged co-residence of two ~30B NVFP4 models
(vLLM primary + fallback) + Chatterbox TTS + Parakeet STT results in a contended
GPU and fragmented CUDA memory. After several hours or under sustained load,
Parakeet's CUDA context becomes stale, and new transcription requests fail deep in
NeMo's initialization.

**Fix:**

Restart the STT container to clear the stale CUDA context:

```bash
docker restart model-gear-stt
```

Or cycle the entire fleet:

```bash
lobes fleet down --apply && lobes fleet up --apply
```

Watch `nvidia-smi` to confirm memory is freed before the STT container restarts.

**Diagnosis (if it recurs):**

- Compare GPU memory before/after `docker restart model-gear-stt`.
- Run `nvidia-smi` to check for fragmentation or orphaned allocations.
- If CUDA errors persist, contact NVIDIA support or consider (a) running audio
  on a dedicated GPU, (b) reducing the fleet to a single LLM, or (c) lowering
  `PRIMARY_GPU_MEM_UTIL` and `FALLBACK_GPU_MEM_UTIL` to reduce baseline memory
  load.

Root cause diagnosis is open; see issues #39 and #40 if this resurfaces.

## The `/v1/realtime` WebSocket session (issue #149)

**Status: VALIDATED on the DGX Spark GB10, 2026-07-21** — transcript:
[`docs/evidence/2026-07-21-accept-realtime-spark.txt`](evidence/2026-07-21-accept-realtime-spark.txt).
A live run drove a full session through the gateway tunnel against the real
Silero model and the real Parakeet and Chatterbox sidecars: `session.created`
→ `speech_started` → `speech_stopped` → transcription, all on one connection,
at **both** wire rates (24000 Hz and the 16000 Hz passthrough), plus the 401
on an unauthenticated handshake and the 426 on a plain GET.

Four things stay **UNVALIDATED** and must not be claimed: a real
**microphone** (every live run used synthesized Chatterbox audio, so
reachy-mini-cli's mic path is still unproven end to end), the
**VAD-unavailable** error path, **concurrent** sessions, and the
**max-turn** force-commit — the last three are covered offline only. The
live smoke procedure that will produce one — connect, stream, boundaries,
transcript, all over one connection, plain-`websocket-client`-level like
`scripts/audio-smoke.py` — is issue #149's task t8. Until that transcript
lands, treat every claim below as offline-proven, not live-validated.

### The IOUs this redeems

- `lobes/realtime/app.py`'s own module docstring used to read "PR2 adds the
  `/v1/realtime` WebSocket route" as a forward promise. The route now exists
  (`@app.websocket("/v1/realtime")`), wiring the stdlib-tested
  `_session.py` / `_segmenter.py` / `_pcm.py` modules to a real Silero model
  and real scipy resampling.
- This doc's own Boundary section (below) used to say the WebSocket protocol
  "does not change... that is planned for a later release." That was the
  **#149 baseline probe**: the deployed realtime container served four batch
  routes (`/health`, `/v1/health/ready`, `/v1/audio/transcriptions`,
  `/v1/audio/speech`) and no WebSocket at all — which is why reachy-mini-cli
  had to endpoint client-side with an energy threshold (measured failure: a
  five-word question arriving as the fragment "Ready, she"). The route below
  redeems that IOU; the Boundary section states only what is still true.

### Reachability

Served **through the gateway**, not the bridge port directly. The gateway's
`GET /v1/realtime` handler (`lobes/gateway/server.py::_handle_realtime`, via
`lobes/gateway/_realtime.py::plan_realtime_upgrade`) relays the WebSocket
101-upgrade handshake to the local `realtime` bridge, then pumps opaque bytes
both directions until either side closes (`run_tunnel` / `pump`) — the
gateway never parses the WebSocket protocol itself, only the HTTP handshake.
The same opt-in `GATEWAY_API_KEY` bearer check gates the handshake exactly
like every other `/v1/*` data-plane route: a missing or wrong key is
rejected before any tunnel or session is allocated. A plain `GET
/v1/realtime` (no `Upgrade: websocket` header) gets **426** ("send an
Upgrade: websocket handshake"), not a 404 — the route exists, it just was not
asked for correctly. A declared-off `stt` lane (`STT_FEASIBLE=false`) gets
the same **404 `role_infeasible`** the batch STT route gets, naming
`hosted_by` when a peer origin is declared.

### Connect URL and session config

Session config is **connect-URL query parameters**, not a first WS message:

```text
wss://<gateway>/v1/realtime?input_sample_rate=16000
```

| Param | Default | Accepted |
|---|---|---|
| `input_audio_format` | `pcm16` | `pcm16` only |
| `input_sample_rate` | `24000` | `24000` or `16000` |
| `input_channels` | `1` | `1` (mono) only |
| `turn_detection` | `server_vad` | `server_vad` only |
| `aec_mode` | `none` | `none` or `aec` |

Wire format is **PCM16 mono little-endian**, streamed as **binary** WebSocket
frames at whatever chunking granularity the client sends — the server
reassembles the stream; a frame need not align to a whole sample, let alone a
whole 32 ms VAD chunk (`lobes/realtime/_pcm.py::take_aligned_samples`).
`input_sample_rate` defaults to **24000 Hz** (OpenAI-Realtime-compatible);
**16000 Hz is also accepted** (Parakeet/Silero's native rate — the server
skips resampling entirely in that case, see `_pcm.py::needs_resample`). Any
other rate is rejected as an invalid session config, and the socket is
closed (WS code 1008) before any audio is accepted — no session is allocated
for a rejected config. The server resamples 24 kHz down to 16 kHz itself,
server-side, via scipy (`lobes/realtime/app.py::_resample_to_16k`) — the
client never resamples.

### Event flow

Events come back as JSON **text** frames (schema: `lobes/realtime/_session.py`,
`EventType`):

1. `session.created` — sent immediately after the handshake, confirming the
   negotiated config including the resolved `input_sample_rate`. A client
   that sent no query params at all can read the effective defaults off
   this event.
2. `input_audio_buffer.speech_started` — server-side Silero VAD crossed
   `VAD_THRESHOLD` on a chunk; the turn's audio begins with up to
   `VAD_PREFIX_PADDING_MS` of pre-roll so the syllable before detection is
   never lost.
3. `input_audio_buffer.speech_stopped` — the turn committed, either because
   `VAD_SILENCE_MS` of continuous non-speech confirmed the stop
   (`reason="silence"`), or the max-turn cap fired (`reason="max_turn"` —
   see below).
4. `conversation.item.input_audio_transcription.completed` — the committed
   turn's audio was forwarded to `settings.stt_url` (Parakeet — the exact
   same backend and WAV-wrapping the batch `/v1/audio/transcriptions` route
   uses) and transcribed, **on the same connection** — no separate batch
   call.
5. `error` — a documented `ErrorCode`, never a bare exception string:
   `invalid_session_config` (bad config, rejected before any session
   exists), `vad_unavailable` (Silero failed to load, or a later VAD call
   raised — **distinct from ordinary silence, which emits no event at
   all**), `stt_forward_failed` (the committed turn's Parakeet forward
   failed: unreachable backend, non-2xx, non-JSON, or a body missing
   `text` — **a turn is never silently dropped**).

This is audio-in only: the route never sends audio back over `/v1/realtime`
— no `response.create`, no LLM turns, no TTS-out on this connection (the
full OpenAI Realtime conversation surface is an explicit non-goal; TTS stays
the batch `/v1/audio/speech` route).

### Max-turn cap: force-commit, not an error

A stream that never falls silent (a stuck mic, an uninterrupted monologue)
would otherwise grow one turn's buffered audio without bound.
`VAD_MAX_TURN_MS` (default 30000 ms; env-tunable — see
`docker-compose.audio.yml` / `env.audio.example`) bounds it: once a turn's
accumulated audio reaches the cap, the segmenter **force-commits** it as an
ordinary `input_audio_buffer.speech_stopped` event with `reason="max_turn"`
— **this is not an error and never raises** — and the session proceeds
straight to the transcription forward, same as a silence-committed turn. A
consumer must not expect an `error` event on this path; inspect `reason` if
you want error-like handling of an unusually long turn.

### Ephemeral sessions — the restart contract

There is no resume. `Session.teardown()` (`lobes/realtime/_session.py`)
releases every session's bookkeeping from **any** state (idle, mid-speech,
mid-transcription) on a disconnect for **any** reason — client close,
network drop, or the server closing the connection itself after a
`vad_unavailable` error. Nothing here persists to disk. A reconnecting
client always starts a **brand-new session with a brand-new session id** —
there is no state to restore, on either the bridge or the gateway (a dropped
client unwinds both tunnel pump threads on the gateway side,
`lobes/gateway/_realtime.py::pump` / `run_tunnel`). The client contract is:
reconnect and restart the turn you were mid-way through — there is no
partial-turn recovery across a disconnect.

### Talking to it: `scripts/realtime-voice-loop.py`

`/v1/realtime` is **ears only** — audio in, boundaries and transcripts out. A
spoken *conversation* is therefore a client-side composition of three
endpoints this fleet already serves:

| role | endpoint | backend |
|---|---|---|
| ears | `ws /v1/realtime` | Silero VAD + Parakeet |
| brain | `POST /v1/chat/completions` | any generate lane |
| mouth | `POST /v1/audio/speech` | Chatterbox |

`scripts/realtime-voice-loop.py` is that composition, and doubles as the
richest live test of the realtime surface — it exercises a long-lived duplex
session in a way the one-shot `realtime-smoke.py` cannot.

```bash
export LOBES_API_KEY=...        # never pass a key in argv: /proc is world-readable
python3 scripts/realtime-voice-loop.py \
    --device hw:1,0 --channels 2 \
    --sink alsa_output.platform-NVDA2014_00.hdmi-stereo
```

Three behaviours are deliberate, and each was learned the hard way on live
hardware:

- **It answers `PING` with `PONG`.** uvicorn pings roughly every 20 s and
  closes a peer that never pongs. A duplex client that ignores pings dies
  after tens of seconds for no visible reason; a one-shot smoke run finishes
  inside a single ping interval and never notices. If you write your own
  client, handle `OPCODE_PING`.
- **It is half-duplex — no barge-in.** The mic is muted (silence is streamed
  in its place) for the whole synthesize-and-play window, because without
  echo cancellation the mic hears the speakers and the session transcribes the
  machine talking to itself. Real barge-in needs AEC and is tracked in
  [#151](https://github.com/agentculture/lobes-cli/issues/151).
- **It defaults to the Gemma 4 12B lane** (`--model multimodal`), not
  `cortex`. Measured on the DGX Spark: ~1 s to a short reply with no reasoning
  trace. In a spoken turn latency *is* dead air, so speed beats depth; a
  thinking model spends its budget on a trace nobody hears.

`--sink` matters on a box where something else owns the audio device: on the
Spark, `reachy-mini-dae` holds the Reachy speaker exclusively and PipeWire
cannot reach it while that daemon runs, so playback goes to the HDMI sink
instead.

## Boundary / non-goals

The audio surface **does not**:

- Proxy the `/v1/realtime` WebSocket cross-box. The session ships (see
  above) and is served through the gateway on this box only — the #129
  proxy-lobes forwarder is POST-only, so a declared-off `stt` lane 404s the
  handshake `role_infeasible` (naming `hosted_by` when a peer origin is
  declared) rather than tunneling the WebSocket to a peer.
- Enable AEC by default. `aec_mode` defaults to `none` and stays off unless a
  session's connect-URL explicitly requests `aec_mode=aec` — Reachy Mini's
  mic array cancels echo in firmware, so server-side AEC is opt-in, never
  assumed.
- Expose a full OpenAI Realtime conversation surface. `/v1/realtime` is
  audio-in, boundaries, and transcription only — no `response.create`, no
  LLM turns, no TTS-out on the session (see [Event flow](#event-flow) above).
- Swap the STT engine — Parakeet (NeMo ASR) remains the hardcoded STT backend.
  TTS has been migrated from Magpie (NVIDIA NIM, proprietary) to Chatterbox
  (Resemble AI, open-weights, Apache-2.0). Silero VAD is likewise hardcoded —
  none of the three (Parakeet, Chatterbox, Silero) is in the switchable
  catalog (`lobes/catalog.py`).
- Add an audio-specific auth scheme. Both the batch routes and the
  `/v1/realtime` handshake are gated by the same opt-in `GATEWAY_API_KEY`
  bearer check as every other gateway data-plane route — see
  [`docs/gateway-fleet.md#auth-opt-in-bearer-gate`](gateway-fleet.md#auth-opt-in-bearer-gate)
  and [`docs/openai-api.md#fleet-gateway`](openai-api.md#fleet-gateway).

## Memory (co-residence risk)

On a GB10 shared with other services, two ~30B NVFP4 models barely co-fit with
usable KV caches. Adding Parakeet + Chatterbox increases contention. Options:

- Run audio on a **dedicated GPU** (recommended).
- Reduce the fleet to a **single LLM** and use `lobes switch` instead of fleet.
- Tune `PRIMARY_GPU_MEM_UTIL` and `FALLBACK_GPU_MEM_UTIL` in `.env` to lower
  the baseline (the defaults are estimates for a dedicated box).

See [`docs/gateway-fleet.md`](gateway-fleet.md#memory-both-warm) for full memory
guidance and [`docs/gateway-fleet.md#live-validation-findings`](gateway-fleet.md#live-validation-findings--dgx-spark-gb10-2026-05-30)
for concrete measurements on the shared DGX Spark.

## Smoke test

Run the live audio smoke test to verify the stack is serving:

```bash
python3 scripts/audio-smoke.py
# or with a custom base URL (default http://localhost:8080):
python3 scripts/audio-smoke.py --base-url http://10.0.0.42:8080
```

The script:

1. Checks that `GET /openapi.json` lists both `/v1/audio/transcriptions` and
   `/v1/audio/speech`.
2. Generates a 2-second 440 Hz tone (16 kHz, mono, PCM16 WAV).
3. Sends it to `/v1/audio/transcriptions` and confirms a 200 response with a
   `text` field.
4. Prints PASS/FAIL for each step and exits non-zero on failure.

It can also run the **TTS → STT round-trip** check (`check_round_trip`): synthesize
a known phrase through the Chatterbox sidecar, wrap the returned PCM in a 24 kHz
WAV, post it to Parakeet, and assert the transcript echoes the input. This is the
functional proof that the two audio backends actually work together end-to-end:

```bash
python3 scripts/audio-smoke.py \
  --chatterbox-url http://localhost:9100 \
  --stt-url http://localhost:9002
```

This requires a **live GPU box** with `lobes fleet up` already running; it is not
an offline CI test. It reproduces the issue #39 symptom to confirm the fix.
