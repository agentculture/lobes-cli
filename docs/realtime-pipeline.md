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

## The `/v1/realtime` WebSocket session (issues #149, #151)

**Two releases, two very different validation states. Read this first.**

**The session mechanism (#149) is VALIDATED** on the DGX Spark GB10,
2026-07-21 — transcript:
[`docs/evidence/2026-07-21-accept-realtime-spark.txt`](evidence/2026-07-21-accept-realtime-spark.txt).
A live run drove a full session through the gateway tunnel against the real
Silero model and the real Parakeet and Chatterbox sidecars: `session.created`
→ `speech_started` → `speech_stopped` → transcription, all on one connection,
at **both** wire rates (24000 Hz and the 16000 Hz passthrough), plus the 401
on an unauthenticated handshake and the 426 on a plain GET.

**Everything issue #151 adds is DECLARED, not validated.** The base64 event
wire, the opt-in conversation surface (server-side generate + TTS streamed
back on the same socket), barge-in, the per-stage deadlines and the browser
harness under `site/` are **offline-proven only**: every decision lives in
stdlib modules the offline suite covers end to end
(`lobes/realtime/_wire.py`, `_floor.py`, `_turn.py`, `_conversation.py`,
`_session.py`), and `app.py` stays the `pragma: no cover` shell that pumps
them. Per the **#108 evidence rule**, nothing here says *validated* until a
live acceptance transcript lands under `docs/evidence/` — the run that would
produce one (speak, hear the reply, interrupt it mid-playback, through the
documented `ssh -L` flow with a real microphone) is issue #151's task t17.
Note too that the #149 transcript above was recorded against the **raw-binary
input wire #151 has since replaced**: it proves the gateway tunnel, Silero
segmentation and the Parakeet forward, *not* the wire the server speaks
today.

Four things #149 left **UNVALIDATED** stay that way — this work retires none
of them: a real **microphone** (every live run so far used synthesized
Chatterbox audio, so reachy-mini-cli's mic path is still unproven end to
end), the **VAD-unavailable** error path, **concurrent** sessions, and the
**max-turn** force-commit. The `site/` harness is the real-microphone test
vehicle by construction, so t17's run is the one that could retire *that*
item; concurrency stays open regardless (a single-operator browser cannot
prove it), which is also why `TTS_VOICE_CONCURRENCY` ships defaulted to `1`.

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
- This doc used to carry a second, narrower IOU naming **issue #151** by
  number: "It is half-duplex — no barge-in… Real barge-in needs AEC and is
  tracked in #151." That described `scripts/realtime-voice-loop.py` as
  shipped in commit `f1e6ffa`, and its own docstring said the same in the
  same words — a client-side stitch of three endpoints that muted its
  microphone for the whole synthesize-and-play window, because without echo
  cancellation the session transcribed the machine talking to itself. The
  server side matched: **#149's spec recorded no-`response.create` /
  no-LLM-turn / no-TTS-out as non-goal c15**, with the `barge_in_*` and
  response-id machinery shipped but deliberately dormant. #151 is the planned
  lift of exactly that non-goal — it moves the loop **server-side** and makes
  the interruption a first-class session event; see
  [Conversation is opt-in](#conversation-is-opt-in-responsecreate) and
  [Barge-in](#barge-in-speaking-over-the-machine) below. The half-duplex
  framing survives here only as *history* — the script itself is unchanged in
  that respect and remains the pre-#151 fallback.

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
| `system_prompt` | *(env `DEFAULT_SYSTEM_PROMPT`, else the built-in)* | any string (issue #151) |

Audio is still **PCM16 mono little-endian**, and `input_sample_rate` still
defaults to **24000 Hz** with **16000 Hz also accepted** (Parakeet/Silero's
native rate — the server skips resampling entirely in that case, see
`_pcm.py::needs_resample`). Any other rate is rejected as an invalid session
config, and the socket is closed (WS code 1008) before any audio is accepted
— no session is allocated for a rejected config. The server resamples 24 kHz
down to 16 kHz itself, server-side, via scipy
(`lobes/realtime/app.py::_resample_to_16k`) — the client never resamples.

What changed in #151 is how those bytes are *framed*. See next.

### The wire is base64 JSON events, in both directions (issue #151)

**This is a breaking change, deliberately taken.** #149 shipped raw **binary**
WebSocket frames as the input wire. #151 supersedes it: the wire is
OpenAI-Realtime-shaped **JSON text events carrying base64 audio**, in both
directions.

| direction | event | payload |
|---|---|---|
| client → server | `input_audio_buffer.append` | `{"type": ..., "audio": "<base64 PCM16>"}` |
| server → client | `response.audio.delta` | `{"type": ..., "response_id": ..., "delta": "<base64 PCM16>"}` |

Chunking granularity is still the client's choice — an append event's audio
need not align to a whole sample, let alone a whole 32 ms VAD chunk; the
server reassembles the stream (`lobes/realtime/_pcm.py::take_aligned_samples`).
Outbound deltas are 100 ms frames (`_wire.py::DEFAULT_DELTA_CHUNK_BYTES` —
4800 bytes at 24 kHz, the single source of truth for outbound chunk size in
the tree), small enough that the reply starts quickly and an interruption
discards only a bounded remainder.

A raw binary frame is **no longer read as audio**. One arriving now yields
the named `error` / `invalid_wire_event` event
(`_wire.py::WireErrorCode.UNSUPPORTED_FRAME_TYPE` in the message text) and
the session stays open — never a silent misread, and never a compatibility
path the codec preserves. The codec is `lobes/realtime/_wire.py`
(`decide_inbound_message` is the single classification point `app.py` calls
per received message), and it is stdlib-only, so the framing arithmetic is
unit-tested with none of the `[realtime]` extra installed.

**Migrating a client.** Where you sent `ws.send_binary(pcm)`, send
`ws.send(json.dumps({"type": "input_audio_buffer.append", "audio":
base64.b64encode(pcm).decode()}))` instead; where you read binary frames
back, read `response.audio.delta` events and base64-decode `delta`. The
in-repo clients (`scripts/realtime-smoke.py`, `scripts/realtime-voice-loop.py`)
already speak this wire and are the worked reference — including the duplex
survival rules a non-browser client still needs (pong uvicorn's pings, use
`select()`-based read deadlines, guard writes with a lock).

**The coordinated break with reachy-mini-cli.** The deployed robot streams
the #149 binary wire and **cannot stream against this server until it
adapts** —
tracked as **reachy-mini-cli#115**. This is a recorded, operator-accepted
decision (issue #151, frame decision c40), not a regression discovered after
the fact: sequencing the reachy commit against the lobes deploy is the
operator's call, and the window where an un-updated robot is mute is the
accepted cost of having exactly one wire instead of two. What reachy does
*not* have to change is its behaviour — see the ears-only default below.

### Event flow — the default (listening) sequence

Events come back as JSON **text** frames (schema: `lobes/realtime/_session.py`,
`EventType`). A session that never opts into conversation emits exactly this
sequence and nothing else — that is the contract reachy-mini-cli depends on,
and it is a structural property of `_conversation.py`, not a promise: every
floor call in that module sits behind an `if self.armed` guard.

1. `session.created` — sent immediately after the handshake, confirming the
   negotiated config including the resolved `input_sample_rate`. A client
   that sent no query params at all can read the effective defaults off
   this event.
2. `input_audio_buffer.speech_started` — server-side Silero VAD crossed
   `VAD_THRESHOLD` on a chunk; the turn's audio begins with up to
   `VAD_PREFIX_PADDING_MS` of pre-roll so the syllable before detection is
   never lost. Carries **`at_ms`** (see below).
3. `input_audio_buffer.speech_stopped` — the turn committed, either because
   `VAD_SILENCE_MS` of continuous non-speech confirmed the stop
   (`reason="silence"`), or the max-turn cap fired (`reason="max_turn"` —
   see below). Carries **`at_ms`** and **`reason`**.
4. `conversation.item.input_audio_transcription.completed` — the committed
   turn's audio was forwarded to `settings.stt_url` (Parakeet — the exact
   same backend and WAV-wrapping the batch `/v1/audio/transcriptions` route
   uses) and transcribed, **on the same connection** — no separate batch
   call.
5. `error` — a documented `ErrorCode`, never a bare exception string:
   `invalid_session_config` (bad config, rejected before any session
   exists), `vad_unavailable` (Silero failed to load, or a later VAD call
   raised — **distinct from ordinary silence, which emits no event at
   all**), `invalid_wire_event` (a malformed client frame — bad JSON, a bad
   base64 `audio` field, or a raw binary frame; the specific
   `WireErrorCode` reason is named in the message text and the session stays
   open), `stt_forward_failed` (the committed turn's Parakeet forward
   failed: unreachable backend, non-2xx, non-JSON, or a body missing
   `text` — **a turn is never silently dropped**).

**Boundary events now carry `at_ms` and `reason` (issue #151).** Both are the
segmenter's own values, and both were computed and then dropped before the
wire until #151 threaded them through. `at_ms` is elapsed, 32 ms-quantised
**audio-stream** time — a different clock from the event's `timestamp_ms`
(a monotonic process clock), never mixed with it. Two things follow, and both
are the point:

- **VAD tuning becomes observable against a live session.** The effect of
  `VAD_THRESHOLD` / `VAD_SILENCE_MS` / `VAD_PREFIX_PADDING_MS` was previously
  visible only in offline fixture replay; an operator now reads boundary
  timings straight off the live event stream (the `site/` harness renders
  exactly this).
- **A client can finally distinguish a force-commit from a silence-confirmed
  stop.** This doc has always described the `max_turn` force-commit as "a
  normal boundary event, not an error" — true of the server's *behaviour*
  all along, but until `reason` reached the wire **no client could actually
  observe the difference**. The distinction was a server-side fact the
  protocol did not express. Now it does.

### Conversation is opt-in (`response.create`)

**Ears-only is the default and stays the default.** A session is *listening
only* until the client sends a `response.create` event; one that never sends
it gets the sequence above and nothing more. Arming is session-level and
idempotent — send it once at connect and every committed turn thereafter is
answered, or send it after each transcript, OpenAI-style. A transcript the
floor did not take is remembered as *pending* and answered by the next
trigger, and cleared once answered, so a duplicate trigger cannot produce two
replies to one turn.

Once armed, a committed turn becomes a spoken reply **on the same
connection**, with no second HTTP call from the client:

1. `response.created` — the machine took the floor.
2. the transcript plus the session's history and system prompt are POSTed to
   the generate lane at `OPENAI_BASE_URL` (default `http://gateway:8000`,
   i.e. the fleet's own gateway — no extra vLLM), with
   `chat_template_kwargs {"enable_thinking": false}`.
3. `response.text.done` — the reply text, verbatim.
4. `response.audio.delta` × N — Chatterbox's PCM16, base64, 100 ms per event.
   Chatterbox emits **24 kHz**, which `protocol.py` pins equal to the client
   wire rate, so **audio-out never resamples**: the bytes are a passthrough
   plus one base64 encode.
5. `response.done` — the reply was delivered in full; the floor returns to
   the caller.

**The voice lane defaults to `multimodal`**, not `cortex` — the same measured
reason `scripts/realtime-voice-loop.py` records in-tree: the Gemma 4 12B lane
answers a short spoken turn in about a second, where the 27B thinking lane
spends that budget on a reasoning trace nobody hears. In a spoken turn,
latency *is* dead air. Override with `OPENAI_MODEL` (e.g. `cortex` for full
reasoning at the cost of latency). If the configured lane is not hosted on
this box, the gateway's `404 role_infeasible` becomes the named
`generate_failed` error event **carrying the `hosted_by` peer hint** — never
a silent fallback to another lane, which is the whole point of honest
referral.

**History and the system prompt live on the server now** (they used to live
in the voice-loop client). Both are per-session and in-memory only: no disk,
no module state, and `Session.teardown()` drops them with everything else —
consistent with the ephemeral-session contract below. The prompt resolves in
three layers: the connect-URL `system_prompt` param wins, else the
operator's `DEFAULT_SYSTEM_PROMPT` env value, else the built-in spoken-style
default in `_session.py`. That default is load-bearing, not decoration:
Chatterbox reads the reply **verbatim**, so a prompt that lets the model
revert to its written register comes back as literal asterisks and list
markers read aloud.

**Every stage has a deadline.** `transcribe`, `generate` and `tts` each get a
bounded wait (60 s each, mirroring `app.py`'s own forward timeouts and
`tts_client`'s httpx read timeout). On expiry the floor **returns to the
caller** with the named `response_timeout` error — with the stage named in
the message text, since one code covers all three — rather than leaving the
session wedged mid-response. Deadlines expire in a watchdog tick, so an
answer that arrives first simply wins; a completion belonging to a turn the
floor has already left is ignored rather than spoken over a later turn.

**Voice TTS no longer queues behind batch TTS.** `tts_client.py` used to gate
every Chatterbox request behind one shared semaphore, so a spoken reply could
wait on unrelated `POST /v1/audio/speech` work — dead air, in a conversation.
The lanes are now two independent pools: `TTS_CONCURRENCY` gates the batch
lane only (unchanged), `TTS_VOICE_CONCURRENCY` gates the voice lane only.
Splitting the pool is a structural guarantee; raising a shared ceiling would
only have been a probabilistic one. Both default to `1` — the voice lane
deliberately claims lane *isolation*, which is proven, and not multi-session
throughput, which is not (concurrent sessions remain unvalidated).

### Barge-in: speaking over the machine

Speech detected while the machine holds the floor is a **barge-in**, and it
cancels the reply in flight:

- both abandonment hooks fire — `cancel_generate()` **and** `cancel_tts()`,
  from every state. The floor cannot know whether the route had already handed
  off to TTS when the onset landed, and both hooks are idempotent, so
  cancelling both closes that race by construction rather than by timing;
- the **undelivered remainder is never sent** — that is what pumped,
  chunk-at-a-time delivery buys. A single blocking "send it all" would leave
  nothing to interrupt;
- exactly one `response.interrupted` event goes out, carrying the truncation
  marker `truncated: true`. The client stops its own local playback of
  already-delivered frames when it sees it; both halves — server-side
  truncation and client-side stop — are required for an interruption to feel
  instant;
- the floor returns to the user.

**Only what was plausibly heard enters history.** The server estimates the
spoken prefix from how much audio actually went out and records *that*, not
the whole reply (`_floor.py::estimate_spoken_prefix`). Recording the full
text would be the worse lie: the next turn's context would claim the machine
said things the user cut off before hearing. Nothing delivered means nothing
heard, and history records nothing. It is an estimate, not an alignment —
Chatterbox returns audio with no word timings.

The segmenter needed **zero changes** for this: it is a floor-agnostic pure
state machine that never stops segmenting, and has no idea a response is in
flight. A speech onset during playback *is* the trigger; consuming it is the
new floor machine's job (`lobes/realtime/_floor.py`).

`BARGE_IN_WINDOW_MS` (default 750) is armed as a **guard window**, not a
delay: an onset landing less than that long after the machine took the floor
is ignored — no event, no cancel — because the likeliest source of speech in
that instant is the tail of the user's own turn or an echo blip as playback
starts, not a deliberate interruption. It is deliberately wider than
`VAD_SILENCE_MS`'s own 600 ms so ordinary boundary noise inside a reply
cannot double as an accidental interrupt. A **committed turn** arriving while
the machine holds the floor also interrupts (once past the guard) — a turn
that survived the VAD's own silence confirmation is far stronger evidence
than a bare onset, and dropping it would silently discard something the user
said.

`BARGE_IN_MODEL` is read from env and threaded end to end but **consumed by
nothing** — window-only barge-in is what ships. The knob stays declared and
unconsumed until a live run shows the window alone is insufficient; that is
the recorded mitigation, not a shipped feature.

### Muting: a narrowed ban, not a lifted one (deviation d1)

The old half-duplex loop muted its microphone whenever it spoke. That was an
**AEC substitute**, and it is exactly why barge-in was impossible: you cannot
interrupt a machine that has stopped listening. So the rule is not "no
muting" — it is sharper than that, and the distinction is the whole design:

- **Automatic mute-during-playback remains FORBIDDEN.** No client of this
  session may mute, gain-zero, or otherwise deafen its capture path *in
  reaction to* a playback or response event. The `site/` harness enforces
  this as a build-time grep gate over an explicitly marked forbidden zone
  (`site/src/scripts/no-mic-mute.test.ts`), not as a convention.
- **User-initiated mute and mic-off are ALLOWED** (approved deviation `d1`,
  recorded in `.devague/deliveries/`). Real hardware exists now: the Reachy
  Mini microphone cancels echo in firmware, the browser gets echo
  cancellation from `getUserMedia({audio: {echoCancellation: true}})`, and
  playback lands on Reachy's own speaker or an HDMI monitor. **AEC is
  genuinely owned at the client edge**, so a human pressing mute is a
  privacy and control affordance that does not reintroduce the failure mode
  the ban existed to prevent.

The ban narrowed *because the reason for it moved*, not because it stopped
mattering. A muted stretch must also stay honest in the event stream: the
harness renders a client-origin mute row explicitly, so **muted, silence and
disconnected read as three different nothings** rather than one ambiguous
gap.

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
you want error-like handling of an unusually long turn — and since #151 put
`reason` on the wire, that inspection is finally something a client can
actually do. The cap itself is still **UNVALIDATED** live (covered offline
only).

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

### Talking to it: the browser harness, and the terminal fallback

Two clients drive this surface from the repo.

**`site/` — the local-only Astro harness (issue #151).** A browser page that
opens the session with a real microphone, renders every event type as it
arrives (boundaries with their `at_ms` timings, transcripts, reply text,
audio out, interruptions, and each named error code distinctly), and plays
the reply back. It exists so the surface can be *experienced* rather than
inferred from terminal prints, and it is the real-microphone test vehicle the
acceptance run needs. It is **never deployed** — no workflow publishes it;
CI only builds it so a broken site fails a PR. Because `getUserMedia`
requires a secure context, the browser must reach both the site and the
gateway as `localhost`: the documented flow is `ssh -L` port forwarding from
the operator laptop, with the API key held only by a local
credential-injecting WebSocket proxy and never sent to the browser. See
[`site/README.md`](../site/README.md).

**`scripts/realtime-voice-loop.py` — the pre-#151 terminal fallback.** Before
the conversation surface existed, a spoken conversation had to be a
*client-side* composition of three endpoints this fleet already serves:

| role | endpoint | backend |
|---|---|---|
| ears | `ws /v1/realtime` | Silero VAD + Parakeet |
| brain | `POST /v1/chat/completions` | any generate lane |
| mouth | `POST /v1/audio/speech` | Chatterbox |

That is what this script is (commit `f1e6ffa`), and it still works: its
loop is obsolete-by-default now that the server can run it, but it stays the
non-browser live test and the fallback for a client that has not adopted the
conversation surface. It has been migrated to the base64 event wire like
every other in-repo client; `brain`/`mouth` stay plain HTTP POSTs.

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
- **It is half-duplex, and that is now HISTORY — the reason it is worth
  keeping.** This script mutes its own microphone (streaming silence in its
  place) for the whole synthesize-and-play window, because without echo
  cancellation the mic hears the speakers and the session transcribes the
  machine talking to itself. That mute is why the loop can never be
  interrupted: you cannot barge in on a machine that has stopped listening.
  The server-side surface above does not work this way, and the ban on
  *automatic* mute-during-playback is what keeps it that way — see
  [Muting](#muting-a-narrowed-ban-not-a-lifted-one-deviation-d1). This doc
  used to close that bullet with "real barge-in needs AEC and is tracked in
  #151"; #151 is what you are reading.
- **It defaults to the Gemma 4 12B lane** (`--model multimodal`), not
  `cortex`. Measured on the DGX Spark: ~1 s to a short reply with no reasoning
  trace. In a spoken turn latency *is* dead air, so speed beats depth; a
  thinking model spends its budget on a trace nobody hears. The server-side
  voice lane inherits exactly this default and this reasoning.

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
  session's connect-URL explicitly requests `aec_mode=aec`, and there is no
  server-side DSP behind it — **AEC is owned at the client edge**: Reachy
  Mini's mic array cancels echo in firmware, the browser gets it from
  `getUserMedia` constraints, and a mic-speaker unit may do it in hardware.
  Both known consumers cover it, so the server's `AECMode` stays a declared
  passthrough. That client-edge ownership is precisely what makes barge-in
  possible at all, and what narrowed the mute ban (see
  [Muting](#muting-a-narrowed-ban-not-a-lifted-one-deviation-d1) above).
- Force conversation on anyone. `/v1/realtime` answers only after an explicit
  `response.create`; a session that never sends one is transcription-only,
  byte-for-byte the #149 sequence on the new wire.
- Expose the **full** OpenAI Realtime API. This session adopts the
  **audio-path event shapes only** — `input_audio_buffer.append`,
  `response.audio.delta`, `response.create`, and the transcription events.
  `session.update` semantics, the complete `conversation.item.*` schema, the
  full response lifecycle, tool calls over the session and ephemeral tokens
  are a **named follow-up**, not a gap to read as almost-done: claiming more
  would over-advertise. Nothing here even reads `response.create`'s body.
- Resume anything. Sessions stay ephemeral (below) — an interrupted or
  in-flight response is simply gone on disconnect; there is no replay.
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
