# model-gear owns the audio I/O surface end-to-end: `model fleet up` brings up Parakeet STT plus the OpenAI-compatible `/v1/audio/transcriptions` (and `/v1/audio/speech`) facade on :8080 from model-gear's own vendored realtime app and compose template, with a healthcheck that reflects model readiness — unblocking reachy-mini-cli's 'hey reachy' wake-word (closes #39 + #40).

> model-gear owns the audio I/O surface end-to-end: `model fleet up` brings up Parakeet STT plus the OpenAI-compatible `/v1/audio/transcriptions` (and `/v1/audio/speech`) facade on :8080 from model-gear's own vendored realtime app and compose template, with a healthcheck that reflects model readiness — unblocking reachy-mini-cli's 'hey reachy' wake-word (closes #39 + #40).

## Audience

- reachy-mini-cli (the wake-word client) and any client wanting the documented OpenAI /v1/audio/* surface behind :8080; secondarily model-gear operators running 'model fleet up'

## Before → After

- Before: The live :8080 facade is the realtime-api sibling build exposing only / , /health, /v1/realtime (WS) — 404 on REST /v1/audio/transcriptions. Parakeet (:9002) 500s on every transcription with CUDA error: unknown error after long uptime on the contended GB10, while its healthcheck still reports 'healthy'. model-gear's vendored facade has the REST routes but is not in its compose template, so it never reaches the box.
- After: model-gear's compose template + 'model fleet up' bring up Parakeet STT and the vendored realtime facade; GET :8080/openapi.json lists /v1/audio/transcriptions and /v1/audio/speech; POST a multipart WAV to :8080/v1/audio/transcriptions forwards to Parakeet and returns the OpenAI-shaped {text:...}; Parakeet's healthcheck exercises a real transcription so 'healthy' means actually-serving.

## Why it matters

- reachy-mini-cli Tier-2 wake-word ('hey reachy') is blocked on a working STT endpoint; today the only working STT is raw Parakeet :9002 (when not 500ing), bypassing the OpenAI-compatible facade. Ending the vendored-vs-deployed drift gives one owned, documented audio surface.

## Requirements

- model-gear's packaged compose template (lobes/templates/docker-compose.yml) gains a Parakeet STT service and a realtime-facade service built from the vendored lobes/realtime app, so 'model init' + 'model fleet up' materialise and start them.
  - honesty: 'model fleet up --apply' on a clean box brings up parakeet + facade services healthy, and 'model init' materialises them into the deployment dir without manual compose edits.
- The realtime facade service serves POST /v1/audio/transcriptions (forward to Parakeet) and POST /v1/audio/speech (forward to Magpie) on :8080, and its /openapi.json lists both routes.
  - honesty: curl :8080/openapi.json lists both /v1/audio/transcriptions and /v1/audio/speech, and a multipart WAV POST to /v1/audio/transcriptions returns 200 with {text:...} sourced from Parakeet.
- The Parakeet container's Docker healthcheck exercises real model readiness (a tiny transcription or equivalent model-ready probe), so the container only reports 'healthy' when it can actually transcribe.
  - honesty: When the Parakeet model is loaded and serving, the healthcheck passes; when the model path is broken (e.g. CUDA-unknown 500s), the healthcheck reports unhealthy within its retry window.
- Documentation (a docs/realtime-pipeline.md and/or README 'Audio I/O' section) states that model-gear owns the live :8080 facade, how 'model fleet up' brings up the audio routes, and the runbook for clearing a stale Parakeet CUDA context (restart).
  - honesty: A reader following only the doc can stand up the audio surface and recover a wedged Parakeet without reading source, and the doc names which project owns the live :8080 container.
- model-gear's compose template gains a Magpie TTS service (NGC NIM, NGC_API_KEY from .env) so the facade's /v1/audio/speech route resolves to a fleet-owned backend.
  - honesty: 'model fleet up' brings up Magpie healthy and POST :8080/v1/audio/speech returns audio; 'model init' env.example documents NGC_API_KEY.

## Honesty conditions

- After 'model fleet up', :8080 serves the REST audio routes from model-gear's own vendored app (not the sibling build), verified by /openapi.json + a 200 transcription.
- reachy-mini-cli can point REACHY_STT_URL at the :8080 facade (or :9002) and get a 'hey reachy' transcription back.
- The drift is real and reproducible today: live :8080 /openapi.json shows only / and /health; :9002 transcription 500s with CUDA-unknown.
- Each after-state assertion is independently checkable by a curl/WAV smoke test documented in the realtime-pipeline doc.
- With the facade route live and Parakeet serving, reachy-mini-cli's Tier-2 wake-word path stops returning 404/500.
- The spec touches only the REST audio facade, Parakeet serving, and its healthcheck — it does not modify the /v1/realtime WS protocol, the TTS/STT engines, or gateway auth.
- Every success signal is a concrete, runnable check (curl, WAV POST, healthcheck flip, doc statement) with a pass/fail outcome.

## Success signals

- GET :8080/openapi.json lists /v1/audio/transcriptions and /v1/audio/speech; the issue-#39 repro WAV POST returns 200 with {text:...}; a real 'hey reachy' WAV transcribes to text containing the phrase; Parakeet's Docker healthcheck flips unhealthy when the model can't transcribe; docs state model-gear owns the live :8080 facade and 'model fleet up' brings up the audio routes.

## Scope / boundaries

- Not building a new STT/TTS engine — Parakeet (NeMo) and Magpie stay the backends. Not solving GPU contention generally; the #39 scope is restart-to-clear-context + a readiness-reflecting healthcheck (a documented co-residency memory split is explicitly out of scope here). Not changing the /v1/realtime WebSocket protocol. Not making the gateway auth-aware.

## Non-goals

- Decommissioning the realtime-api sibling project, or migrating its /v1/realtime WebSocket speech-to-speech loop into model-gear, is out of scope — only the REST audio facade + Parakeet STT move under model-gear ownership.

## Decisions

- model-gear takes over ownership of the audio stack (Parakeet + vendored facade wired into model-gear's compose template and 'model fleet up'), rather than reconciling the realtime-api sibling in place or re-vendoring upstream.
- The #39 Parakeet fix depth is restart-to-clear-stale-CUDA-context plus a Docker healthcheck that exercises real model readiness (transcription), not restart-only and not a full documented co-residency memory contract.
- model-gear's fleet brings up the full audio stack end-to-end: Parakeet STT, Magpie TTS, and the realtime facade — so both /v1/audio/transcriptions and /v1/audio/speech are owned (adds the NGC_API_KEY dependency and GB10 memory to the fleet budget).
- Parakeet's readiness healthcheck is a cheap model-ready probe (confirms the NeMo model is loaded and the CUDA context is live), not a full multipart transcription each interval — resolving the flap/GPU-load risk q1.

## Hard questions

- risk: Adding parakeet + facade (and possibly magpie) to model-gear's fleet raises the GB10 co-residency memory pressure that #39 implicates — the very contention that may have wedged Parakeet.
- risk: A healthcheck that runs a real transcription is heavier/slower than a liveness probe and could flap or add GPU load; a cheap model-ready probe may be the safer form.
