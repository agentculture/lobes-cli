# Build Plan — model-gear owns the audio I/O surface end-to-end: `model fleet up` brings up Parakeet STT plus the OpenAI-compatible `/v1/audio/transcriptions` (and `/v1/audio/speech`) facade on :8080 from model-gear's own vendored realtime app and compose template, with a healthcheck that reflects model readiness — unblocking reachy-mini-cli's 'hey reachy' wake-word (closes #39 + #40).

slug: `model-gear-owns-the-audio-i-o-surface-end-to-end-m` · status: `exported` · from frame: `model-gear-owns-the-audio-i-o-surface-end-to-end-m`

> model-gear owns the audio I/O surface end-to-end: `model fleet up` brings up Parakeet STT plus the OpenAI-compatible `/v1/audio/transcriptions` (and `/v1/audio/speech`) facade on :8080 from model-gear's own vendored realtime app and compose template, with a healthcheck that reflects model readiness — unblocking reachy-mini-cli's 'hey reachy' wake-word (closes #39 + #40).

## Tasks

### t1 — Vendor the Parakeet ASR server into model-gear with a cheap model-ready health probe

- covers: c13, h3
- acceptance:
  - lobes/realtime/parakeet_server.py exposes POST /v1/audio/transcriptions and GET /v1/health/ready
  - GET /v1/health/ready returns 200 only when the NeMo model is loaded AND a trivial CUDA tensor op succeeds, else 503 — a cheap probe, NOT a full multipart transcription each interval (decision c16 supersedes the 'real transcription' wording in c4/c7)
  - A unit test stubs the model to assert 200-when-ready and 503-when-model-None/CUDA-fails, with no GPU required in CI

### t2 — Package build Dockerfiles for the facade and Parakeet services under templates/

- depends on: t1
- covers: c11
- acceptance:
  - lobes/templates/Dockerfile.realtime installs the [realtime] extra and runs lobes.realtime.app; Dockerfile.parakeet installs NeMo ASR and copies parakeet_server.py
  - docker build succeeds for both from a clean checkout (smoke-built in the PR or documented as built)

### t3 — Wire the full audio stack into the compose template + env.example (sole owner of docker-compose.yml)

- depends on: t1, t2
- covers: c11, c12, c17, c4, h1, h2, h5, h12
- acceptance:
  - lobes/templates/docker-compose.yml gains parakeet-stt, magpie-tts, and realtime-facade services; the facade env points STT_URL->parakeet and TTS_URL->magpie and binds host :8080
  - env.example documents NGC_API_KEY (Magpie) and the audio service ports; 'model init' materialises them with no manual compose edits
  - The realtime-facade service builds from Dockerfile.realtime (the vendored lobes app), NOT the realtime-api sibling image
  - Parakeet service healthcheck calls GET /v1/health/ready (the cheap probe from t1)

### t4 — Bring the audio services up via 'model fleet up' (gating + ownership)

- depends on: t3
- covers: c1, h1, h5
- acceptance:
  - 'model fleet up --apply' starts parakeet + magpie + facade and they report healthy on a box with the GPU available
  - 'model fleet status' lists the audio services and the facade /health; the audio surface is served from model-gear's deployment dir, not the sibling project
  - A dry-run (no --apply) prints the planned audio services without starting them (mutation-safety default honoured)

### t5 — Author docs/realtime-pipeline.md (ownership, fleet-up, drift, CUDA restart runbook) + README Audio I/O

- covers: c14, h4, c3, h7, c6, h10
- acceptance:
  - docs/realtime-pipeline.md states model-gear owns the live :8080 facade, shows the 'model fleet up' bring-up, and gives the restart runbook for a stale Parakeet CUDA context
  - The doc records the prior drift (sibling :8080 build had only / and /health; :9002 500'd with CUDA-unknown) and the boundary (no /v1/realtime WS change, engines unchanged, no gateway auth)
  - A reader can stand up the audio surface and recover a wedged Parakeet following only the doc, without reading source

### t6 — Smoke-test + acceptance harness for the audio routes (curl/WAV, openapi, healthcheck flip)

- depends on: t3, t4
- covers: c2, c5, c7, h6, h8, h9, h11, h2, h3, h12, c4, c12
- acceptance:
  - A scripts/ smoke test asserts GET :8080/openapi.json lists /v1/audio/transcriptions and /v1/audio/speech, and a multipart WAV POST returns 200 {text:...}
  - The harness reproduces issue #39's WAV repro (200, not 500) and a 'hey reachy' WAV transcribes to text containing the phrase; REACHY_STT_URL=:8080 works for the wake-word client
  - A test asserts the Parakeet healthcheck flips unhealthy when the model can't transcribe and POST :8080/v1/audio/speech returns audio (Magpie up)

## Risks

- [unknown_nonblocking] Parakeet CUDA-unknown root cause is unconfirmed — restart is the working hypothesis; if it recurs under co-residency with vllm+magpie, deeper driver/contention diagnosis is needed (frame v1) (task t6)
- [unknown_nonblocking] How 'model fleet up' gates the audio services (always-on vs a --audio profile/flag) on the shared GB10 memory budget is unsettled — t4 must choose; adding parakeet+magpie raises the very contention #39 implicates (frame v2 + risk q2) (task t4)
- [follow_up] Spec wording reconciliation: c4/c7 say the healthcheck 'exercises a real transcription' but decision c16 supersedes with a cheap model-ready probe — t1/t5 acceptance must reflect the cheap probe, not full transcription (task t1)
