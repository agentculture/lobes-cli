# Audio realtime pipeline: STT + TTS behind the fleet gateway

## Ownership

**model-gear owns the live audio surface** — the OpenAI `/v1/audio/*` facade
deployed as the `realtime` container in the fleet. The realtime bridge ships in
the model-gear wheel (`model_gear/realtime/app.py` and friends) and is built and
managed by `model init --fleet --audio` / `model fleet up --apply`.

This consolidates what used to be a separate `realtime-api` sibling stack. The
same STT and TTS engines (Parakeet ASR + Magpie TTS) are now owned and versioned
here.

The overlay compose file is **`model_gear/templates/fleet/docker-compose.audio.yml`**.
It is layered on top of the base fleet automatically when present during `model
fleet up`.

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
           └─ POST /v1/audio/speech → [model-gear-tts] :9000 (Magpie NIM)

Both STT and TTS share the GPU with the two LLM backends.
```

## Bring-up

### Prerequisites

- NGC API key for Magpie TTS (get it at [https://org.ngc.nvidia.com](https://org.ngc.nvidia.com) → Setup).
- A GPU box (DGX Spark or dedicated; two ~30B models + audio barely co-fit on a
  shared GB10 — see [Memory and co-residence risk](#memory-co-residence-risk) below).

### Steps

```bash
# 1. Initialize the fleet with the audio overlay
model init --fleet --audio --apply

# 2. Set NGC_API_KEY in the .env file
# Locate ~/.model-gear/.env (or your --compose-dir) and edit:
#   NGC_API_KEY=<your-key>

# 3. Bring up the full audio stack (dry-run by default; --apply commits)
model fleet up --apply

# 4. Check status
model fleet status
```

Each `model init` and `model fleet` verb defaults to **dry-run**; omit `--apply`
to see what would happen, or add `--apply` to execute. This ensures safe-by-default
operation (useful when agents call CLIs in loops).

To customize the compose dir (default `$MODEL_GEAR_DIR` or `~/.model-gear`):

```bash
model init --fleet --audio --compose-dir /path/to/deployment --apply
model fleet up --compose-dir /path/to/deployment --apply
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

- model-gear now owns the audio surface. `model init --fleet --audio` scaffolds
  the complete overlay (compose file, Dockerfile for realtime and Parakeet,
  env keys), and `model fleet up --apply` builds and starts all three services
  (`tts`, `stt`, `realtime`) behind the gateway.
- The realtime bridge forwards `/v1/audio/transcriptions` and `/v1/audio/speech`
  to the backends (Parakeet and Magpie respectively) and wraps their responses
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
(vLLM primary + fallback) + Magpie TTS + Parakeet STT results in a contended GPU
and fragmented CUDA memory. After several hours or under sustained load, Parakeet's
CUDA context becomes stale, and new transcription requests fail deep in NeMo's
initialization.

**Fix:**

Restart the STT container to clear the stale CUDA context:

```bash
docker restart model-gear-stt
```

Or cycle the entire fleet:

```bash
model fleet down --apply && model fleet up --apply
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

## Boundary / non-goals

The audio surface **does not**:

- Change the `/v1/realtime` WebSocket protocol (that is planned for a later
  release; the current surface is REST only: `/v1/audio/transcriptions` and
  `/v1/audio/speech`).
- Swap the STT/TTS engines — Parakeet (NeMo ASR) and Magpie (NVIDIA NIM) remain
  the hardcoded backends.
- Make the gateway auth-aware — the same `/v1/chat/completions` gateway token
  that works for the LLM does not yet extend to audio. Plan to add per-endpoint
  auth in a later release.

## Memory (co-residence risk)

On a GB10 shared with other services, two ~30B NVFP4 models barely co-fit with
usable KV caches. Adding Parakeet + Magpie increases contention. Options:

- Run audio on a **dedicated GPU** (recommended).
- Reduce the fleet to a **single LLM** and use `model switch` instead of fleet.
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

This requires a **live GPU box** with `model fleet up` already running; it is not
an offline CI test. It reproduces the issue #39 symptom to confirm the fix.
