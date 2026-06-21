# Chatterbox TTS — realtime audio sidecar

> The TTS backend for model-gear's audio fleet overlay.  Replaces the abandoned
> Magpie NIM (proprietary NVIDIA NIM, deprecated) after a bake-off that chose
> Chatterbox over Orpheus.  No NGC key required — Chatterbox is open-weights
> (Resemble AI, Apache-2.0).

## What it is

- **Chatterbox** by Resemble AI — a 0.5B parameter TTS model.
- **24 kHz mono PCM16** output — matches the OpenAI Realtime API's
  `CLIENT_SAMPLE_RATE`, so no resampling is needed in the bridge.
- One built-in default voice; **zero-shot voice cloning** via a reference `.wav`
  clip (`audio_prompt_path`).
- Key generation API:

  ```python
  from chatterbox.tts import ChatterboxTTS
  model = ChatterboxTTS.from_pretrained(device="cuda")
  wav = model.generate(text, exaggeration=0.5, cfg_weight=0.5)
  # wav is a 1-D torch tensor at model.sr (24000 Hz)
  # for cloning: wav = model.generate(text, audio_prompt_path="ref.wav", ...)
  ```

## Bake-off: Chatterbox vs Orpheus

| Criterion | Chatterbox | Orpheus |
|---|---|---|
| RTF (real-time factor, warm) | **~0.55–0.64** | ~4.0 (transformers fallback) |
| VRAM | **~3.1 GB** | ~6.7 GB |
| Output sample rate | **24 kHz** (matches client) | 24 kHz |
| Cold-load time | ~19 s | ~35 s |
| arm64 / Blackwell (sm_120) | **works** | blocked — vLLM fast path unimplemented on sm_120 |
| NGC key required | no | no |

Orpheus was eliminated because its fast vLLM-based inference path is blocked on
the sm_120 (Blackwell) ISA, so it falls back to the `transformers` path at ~4×
real-time — too slow for the realtime pipeline.  Chatterbox runs faster than
real-time on the same hardware.

## arm64 / aarch64 install recipe (Dockerfile.chatterbox)

Plain `pip install chatterbox-tts` pulls CPU-only torch on aarch64 (PyPI does
not yet ship a GPU arm64 wheel for current PyTorch).  The working recipe:

```bash
# Install the proven cu128 torch first (cu128 wheels run on the cu130 host driver
# via CUDA forward-compatibility):
pip install torch==2.11.0+cu128 torchaudio==2.11.0+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# Install chatterbox without deps to prevent a torch downgrade to CPU:
pip install chatterbox-tts==0.1.7 --no-deps

# Required runtime deps (exact versions validated on the DGX Spark GB10):
pip install "numpy>=1.24,<2.0" librosa==0.11.0 pyloudnorm \
    transformers==5.2.0 diffusers==0.29.0 conformer==0.3.2 \
    safetensors==0.5.3 s3tokenizer spacy-pkuseg pykakasi==2.3.0 omegaconf

# Perth (Resemble AI vocoder) — pinned from PyPI:
pip install resemble-perth==1.0.1 --no-deps
```

Notes:

- **torchcodec is broken on arm64 GB10** (missing `libnppicc`) — do NOT use
  `torchaudio.save`.  Write PCM/WAV with stdlib `wave` instead.
- The `Dockerfile.chatterbox` base image is
  `nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04` (a lean CUDA runtime with **no**
  preinstalled torch), so the explicit cu128 `torch`/`torchaudio` install above is
  required.  The NGC `nvcr.io/nvidia/pytorch` base was rejected — its bundled
  torchaudio ABI conflicts with Perth at runtime.
- Ubuntu 24.04 ships a RECORD-less, PEP-668-marked pip; the Dockerfile bootstraps
  a clean pip via `get-pip.py` after removing the `EXTERNALLY-MANAGED` marker.

## Sidecar HTTP contract

The `chatterbox` service (built by `Dockerfile.chatterbox`) exposes a minimal
FastAPI server (`model_gear/realtime/chatterbox_server.py`):

| Method | Path | Description |
|---|---|---|
| `GET` | `/v1/health/ready` | Liveness/readiness probe → 200 `{"status":"ok"}` |
| `POST` | `/v1/audio/synthesize` | Synthesize text → raw PCM16 mono 24 kHz |

### POST /v1/audio/synthesize

**Request** (JSON body):

```json
{
  "text": "Hello, I am Reachy.",
  "voice": null
}
```

- `text` (required): the utterance to synthesize (non-empty string).
- `voice` (optional):
  - `null` or `""` → built-in default voice.
  - A path ending in `.wav` → zero-shot cloning reference; passed as
    `audio_prompt_path` to `ChatterboxTTS.generate`.

**Response**:

- `Content-Type: audio/pcm`
- Body: raw PCM16 little-endian, mono, **24000 Hz**.
- On error: JSON `{"error": {"message": "..."}}` with an appropriate HTTP status.

The `generate()` call is dispatched off the event loop via `anyio.to_thread.run_sync`
so it does not block the FastAPI worker.

The sidecar is reachable inside the fleet compose network as
`http://chatterbox:${CHATTERBOX_PORT:-9000}`.

## Fleet integration

The `realtime` bridge sets `TTS_URL=http://chatterbox:${CHATTERBOX_PORT:-9000}`;
`tts_client.py` appends `/v1/audio/synthesize` and posts `{"text": ..., "voice": ...}`
as plain JSON (no SSML — Chatterbox does not support SSML).

The `docker-compose.audio.yml` `chatterbox:` service:

```yaml
chatterbox:
  build: { context: ., dockerfile: Dockerfile.chatterbox, args: { MODEL_GEAR_VERSION: ... } }
  restart: unless-stopped
  deploy:
    resources:
      reservations:
        devices:
          - { driver: nvidia, count: all, capabilities: [gpu] }
  expose: ["${CHATTERBOX_PORT:-9000}"]
  volumes:
    - ${HF_CACHE:-~/.cache/huggingface}:/root/.cache/huggingface
  healthcheck:
    test: ["CMD", "python3", "-c", "import urllib.request; urllib.request.urlopen(...)"]
    start_period: 120s
```

Model weights (~3.1 GB) are downloaded from HuggingFace on first boot and cached
in `HF_CACHE` (defaults to `~/.cache/huggingface`).  Cold-load is ~19 s; the
generous `start_period: 120s` covers both container startup and the first
weight-load.

## Voice cloning

Pass the absolute path of a clean mono 16–24 kHz WAV file (3–10 s, single
speaker, low background noise) as `DEFAULT_VOICE` in `.env`:

```bash
DEFAULT_VOICE=/data/voices/speaker.wav
```

The `realtime` bridge passes this through `resolve_voice()` (which returns it
verbatim because it ends in `.wav`) and on to the sidecar's `voice` field, which
the sidecar passes as `audio_prompt_path` to `ChatterboxTTS.generate`.

## Assessment (DGX Spark GB10, 2026-06-21)

Warm inference, single-stream, co-resident with the 27B primary + Parakeet STT.
Validated end-to-end in a real container (`nvidia/cuda:13.0.1`, `--gpus all`) via
a synthesize → Parakeet STT round-trip:

| Metric | Result |
|---|---|
| Real-time factor (RTF, warm) | ~0.55–0.64 |
| VRAM | ~3.1 GB |
| Output sample rate | 24 kHz mono PCM16 |
| Cold model load | ~19 s |
| Default voice quality | natural, human-sounding |
| Zero-shot cloning | supported via `audio_prompt_path` |
