# Parakeet STT — realtime audio sidecar

> The STT backend for lobes's audio fleet overlay.  The speech-to-text half
> of the realtime audio pair; the TTS half is Chatterbox (see
> `docs/chatterbox-tts.md`).  Runs `nvidia/parakeet-tdt-0.6b-v2` via NeMo ASR,
> exposes an OpenAI-shaped `/v1/audio/transcriptions` endpoint, and is fronted by
> the fleet gateway so callers never hit it directly.

## What it is

- **NVIDIA Parakeet TDT** — a 0.6B parameter NeMo ASR model
  (`nvidia/parakeet-tdt-0.6b-v2`).
- **16 kHz mono** input requirement: the server reads any audio format `soundfile`
  supports, resamples to 16 kHz with `scipy.signal.resample` if needed, and
  downmixes multi-channel audio to mono before passing the waveform to NeMo.
- **OpenAI-shaped output** — returns `{"text": "<transcript>"}` on
  `POST /v1/audio/transcriptions`, the same JSON contract as the OpenAI / Riva ASR
  endpoint.  No format translation is needed in the realtime bridge.
- Model name is read from **`PARAKEET_MODEL`** at runtime; default is
  `nvidia/parakeet-tdt-0.6b-v2`.

## Sidecar HTTP contract

The `stt` container (built from `Dockerfile.parakeet`, running `listen_server.py`)
exposes two endpoints:

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/audio/transcriptions` | Multipart upload → transcript JSON |
| `GET` | `/v1/health/ready` | Readiness probe; 503 until model + CUDA are live |

### POST /v1/audio/transcriptions

**Request** — `multipart/form-data`:

- `file` (required): the audio file to transcribe (any format readable by
  `soundfile`; WAV, FLAC, OGG, etc.).
- `language` (optional form field, default `"en"`): passed through for
  forward-compatibility; NeMo Parakeet is English-only and ignores it.

**Response** (200):

```json
{"text": "Hello, I am Reachy."}
```

**Error** — non-200 HTTP status with a JSON body describing the failure.

**Calling the public surface** (through the gateway, default port 8000):

```bash
curl -F file=@clip.wav http://localhost:8000/v1/audio/transcriptions
```

**Calling the sidecar directly** (inside the compose network, for debugging only):

```bash
curl -F file=@clip.wav http://localhost:9002/v1/audio/transcriptions
```

The sidecar is internal-only (`expose:`, not `ports:`); the canonical public path
is always through the gateway.

### GET /v1/health/ready

Returns `200 {"status": "ready"}` once the model is loaded **and** CUDA is live;
otherwise `503 {"status": "not_ready", "reason": "<why>"}` (the reason is
`"model not loaded"` during startup, or `"CUDA not available"` if the CUDA probe
fails).  See the next section for why this matters.

## Readiness (real, not liveness)

The `/v1/health/ready` probe reports ready **only** when two conditions are both
true:

1. **Model loaded** — the NeMo `ASRModel.from_pretrained(...)` call has returned
   and `_model` is non-`None`.
2. **CUDA live** — a trivial tensor op (`torch.zeros(1, device="cuda")` +
   `torch.cuda.synchronize()`) succeeds without raising.

The decision logic lives in `lobes/realtime/_readiness.py`
(`evaluate_readiness`), the single source of truth shared with the realtime
bridge.  The container COPYs a vendored top-level `_readiness.py` alongside
`listen_server.py` so the import resolves inside the container without requiring
the lobes wheel.

A "healthy" Parakeet container is **actually serving** — this is a true readiness
check, not a liveness ping.  The distinction matters: before issue #39 the
healthcheck was liveness-only and reported "healthy" even when transcriptions were
failing with CUDA errors.

Compose healthcheck parameters:

| Parameter | Value |
|---|---|
| `interval` | 30s |
| `timeout` | 10s |
| `retries` | 3 |
| `start_period` | 600s |

The 600 s `start_period` covers both container startup and the slow NeMo model
load (downloading weights + first CUDA initialisation on a shared GPU can take
several minutes on a cold start).

## Fleet integration

Parakeet runs as the **`stt`** service in `docker-compose.audio.yml`, the audio
overlay compose file layered on top of the base fleet:

```yaml
stt:
  build:
    context: .
    dockerfile: Dockerfile.parakeet
  container_name: model-gear-stt
  expose:
    - "${PARAKEET_PORT:-9002}"
  environment:
    - PARAKEET_PORT=${PARAKEET_PORT:-9002}
    - PARAKEET_MODEL=${PARAKEET_MODEL:-nvidia/parakeet-tdt-0.6b-v2}
```

Key points:

- **Port** — `PARAKEET_PORT`, default `9002`.  Internal-only on the compose
  network; reachable inside the fleet as `http://stt:9002`.  The public surface
  is the gateway's `/v1/audio/transcriptions` (default `http://localhost:8000`).
- **Base image** — `scitrera/dgx-spark-vllm:0.16.0-t4`.  Python packages added
  at build time: `nemo_toolkit[asr]`, `soundfile`, `scipy`, and `ffmpeg` (system
  package).
- **Model pre-download** — `Dockerfile.parakeet` pre-fetches
  `nvidia/parakeet-tdt-0.6b-v2` at image build time via `from_pretrained`, so
  first-start latency is model-load only, not download + load.
- **GPU** — claims all NVIDIA devices via `deploy.resources.reservations`; shares
  the GPU with the LLM backends and Chatterbox.
- **Env override** — set `PARAKEET_MODEL` in `.env` to point to a different NeMo
  ASR checkpoint (any checkpoint compatible with `nemo_asr.models.ASRModel`).

The realtime bridge (`model-gear-realtime`) sets
`STT_URL=http://stt:${PARAKEET_PORT:-9002}` and forwards
`/v1/audio/transcriptions` uploads to Parakeet verbatim; the gateway routes all
`/v1/audio/*` traffic to the bridge.

For the full topology diagram and bring-up steps, see
[`docs/realtime-pipeline.md`](realtime-pipeline.md) and
[`docs/gateway-fleet.md`](gateway-fleet.md).

## Stale CUDA context runbook

**Symptom:** transcription requests fail with `500 torch.AcceleratorError: CUDA
error: unknown error` even though `docker ps` shows `model-gear-stt` as
"healthy".

**Cause:** under prolonged co-residence on a shared GB10 (DGX Spark), Parakeet's
CUDA context can go stale while the readiness probe continues to pass (the probe
checks a trivial allocation, not a full inference path).  Root cause is open
(issues #39 / #40).

**Fix:** restart the STT container:

```bash
docker restart model-gear-stt
```

Or cycle the full fleet:

```bash
lobes fleet down --apply && lobes fleet up --apply
```

For diagnosis steps and memory-pressure guidance, see the
[stale CUDA context runbook in `docs/realtime-pipeline.md`](realtime-pipeline.md#runbook-stale-parakeet-cuda-context-cuda-error-unknown-error).

## Not a switchable gear

Parakeet is the **hardcoded STT backend** for the audio overlay — it is not
registered in `lobes/catalog.py` and cannot be selected via `lobes switch`.
The catalog covers generate / embed / score gears only.  The same applies to
Chatterbox TTS.

To use a different ASR model, set `PARAKEET_MODEL` in `.env` to any NeMo ASR
checkpoint compatible with `nemo_asr.models.ASRModel.from_pretrained`.
