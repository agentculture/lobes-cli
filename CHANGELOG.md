# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.26.0] - 2026-06-21

Documentation pass for the realtime audio overlay and the OpenAI API front: a
feature doc per audio backend, a consolidated endpoint reference, and the same
information surfaced through `model learn`, `model explain`, the README, and
`CLAUDE.md`.

### Added

- `docs/parakeet-stt.md`: per-model feature doc for the **Parakeet** STT backend
  (`nvidia/parakeet-tdt-0.6b-v2`, NeMo ASR) — the only audio model that lacked one.
  Covers the HTTP contract, the real (model-loaded + CUDA-live) readiness probe,
  fleet integration, the stale-CUDA-context runbook, and why it is not a switchable
  catalog gear.
- `docs/openai-api.md`: consolidated **OpenAI-compatible API surface** reference —
  every endpoint (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`,
  `/v1/rerank`, `/v1/score`, `/v1/audio/transcriptions`, `/v1/audio/speech`,
  `/v1/models`, `/v1/models/supported`, `/health`), routing semantics (name /
  default / failover / SSE / audio fan-out), per-endpoint `curl` examples, the
  loaded-vs-supported split, and auth/exposure.
- `model explain` topics: `realtime` / `audio` (the `/v1/audio/*` overlay),
  `transcribe` / `stt` / `parakeet` (STT), `speak` / `tts` / `chatterbox` (TTS),
  and `api` / `openai` (the endpoint surface); linked from the explain root.
- `model learn` now documents the realtime audio overlay and the OpenAI API surface
  (text + `--json` `realtime_audio` / `api_surface` fields).
- README sections for **Realtime audio (STT + TTS)** and **The OpenAI-compatible API
  surface**, plus the two audio backends added to the per-model notes.

### Changed

- `CLAUDE.md`: documents the realtime audio overlay alongside the fleet; the CLI
  package tree now lists the `gateway/`, `realtime/`, `explain/`, and `catalog.py`
  surfaces.

### Fixed

- `docs/realtime-pipeline.md`: removed the stale `NGC_API_KEY` bring-up step (a
  Magpie leftover — Chatterbox needs no NGC key) and documented the TTS → STT
  round-trip in `scripts/audio-smoke.py`.

## [0.25.0] - 2026-06-21

### Added

- **Chatterbox TTS sidecar** (`model_gear/realtime/chatterbox_server.py`): a
  FastAPI HTTP server (`GET /v1/health/ready`, `POST /v1/audio/synthesize`) that
  wraps Resemble AI's Chatterbox model and returns raw PCM16 mono 24 kHz audio.
  Supports zero-shot voice cloning via a `.wav` reference path.  Runs as the
  `chatterbox` fleet service built by the new `Dockerfile.chatterbox` (arm64
  cu128 recipe).
- `[chatterbox]` optional-deps group (`fastapi`, `uvicorn`) in `pyproject.toml`.
- `docs/chatterbox-tts.md`: bake-off numbers, arm64 install recipe, sidecar HTTP
  contract, and integration notes.
- `model_gear/templates/fleet/Dockerfile.chatterbox`: arm64 CUDA build — rebased
  on `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` (no preinstalled torch) with
  pinned `torch==2.11.0+cu128` + `torchaudio==2.11.0+cu128` + Perth, fixing the
  NGC pytorch ABI conflict with Perth observed at runtime.
- numpy fast path in `float_tensor_to_pcm16` (stdlib fallback kept for offline CI);
  parity test in `tests/test_chatterbox_pcm16.py`.
- `resolve_voice()` `.wav` check is now case-insensitive (`.WAV` / `.Wav` work).
- Dead SSML code removed from `tts_client.py` (`_insert_ssml_breaks`); stale
  Magpie references updated to Chatterbox across app, client, and tests; speed-ignored
  warning emitted when a non-default speed is passed to `synthesize()`.

### Changed

- **Replaced Magpie TTS with Chatterbox** across the realtime stack:
  `protocol.py` (`TTS_SAMPLE_RATE` 22050→24000, `resolve_voice` rewritten for
  Chatterbox — `.wav` path for cloning, `""` for default), `tts_client.py`
  (plain JSON POST, no SSML/prosody wrapping), `_settings.py` (`tts_url` default
  → `http://chatterbox:9000`, `default_voice` default → `""`),
  `docker-compose.audio.yml` (new `chatterbox:` service replaces `tts:`),
  `env.audio.example` (Magpie/NGC vars removed, `CHATTERBOX_PORT` added).

## [0.24.0] - 2026-06-20

### Added

- **`model overview --live` — a live fleet dashboard.** `overview` was a static
  description; `--live` now probes the running deployment and reports the five
  "what is it doing right now" views: **online** (per-backend health), **offered**
  (served + candidate models, task families, the endpoint list), **busy**
  (in-flight / queued requests), **usage** (cumulative prompt/generation tokens and
  finished requests by reason), and **endpoints**. It is read-only and HTTP-only —
  it works against a local deployment or a `model tunnel` hostname alike, and
  degrades gracefully when a backend or its metrics is unreachable.
- **Gateway `GET /status`** — a model-gear-native JSON aggregate. The fleet's
  backends are internal-only, so the gateway fans out to each one's `/health` +
  `/metrics` and returns `{object: "model-gear.fleet_status", default_model,
  busy: {running, waiting}, backends: [...], endpoints: [...]}`. This is the source
  `model overview --live` reads in the fleet (a bare single-model server is read
  directly from its `/metrics` + `/health`).
- **`model_gear._metrics`** — a small stdlib-only helper that parses vLLM's
  Prometheus `/metrics` (running/waiting, prompt/generation tokens,
  `request_success_total` by finish reason, KV-cache usage) and best-effort HTTP
  probes that never raise.

### Changed

### Fixed

## [0.23.0] - 2026-06-20

### Added

- **Durable vLLM logs that survive restart/recreate (#50).** When a vLLM
  container restarted, its `docker logs` — and any EngineCore crash trace — were
  lost, which blocked root-causing #50 for lack of data. `model init` now
  scaffolds `mg-logwrap.sh`, bind-mounted as each vLLM service's entrypoint: it
  tees stdout+stderr to a per-boot file `<service>-<boot>.log` under a
  host-mounted log dir (`${MODEL_GEAR_LOG_DIR:-<deploy>/logs}` → `/logs/model-gear`),
  then `exec`s the real command so vLLM stays the signal target (graceful
  shutdown) and the exit code (and `restart:` policy) are unchanged. Teeing at the
  process-I/O level captures **both** Python tracebacks and native CUDA/C++ aborts;
  if logging can't be set up it falls back to a plain `exec` and never blocks
  serving. The crash boot is preserved as its own file. Wired into the single-model
  and fleet (`primary`/`embed`/`rerank`) compose templates. See
  `docs/durable-logs.md`.
- **`model logs`** — new read-only verb to list/tail the durable logs, reading the
  host files directly so it works even after the crashed container is gone:
  `model logs` (list boots), `model logs <service>` (tail latest), and
  `model logs <service> --previous` (tail the boot that crashed, after a restart).

### Changed

- `model init` / `model serve` / `model fleet up` pre-create the host log dir
  (user-owned) before compose bind-mounts it, so logs are never root-owned.

### Fixed

## [0.22.1] - 2026-06-19

### Fixed

- **`model fleet status` now reports the embedding + reranker gears.**
  `FLEET_CONTAINERS` listed only `vllm-primary` + `gateway`, so `model fleet
  status` silently omitted the `vllm-embed` / `vllm-rerank` containers the default
  fleet (#44/#47) actually runs. Added `FLEET_EMBED` / `FLEET_RERANK` to the
  default container set — status now lists all four (the opt-in generate fallback
  stays excluded, as it is not in the default compose).

### Changed

- **Aligned the agent/human-facing prose with the co-resident gears (#44/#47).**
  `model learn`, `model overview`, `model explain` (root + fleet), `model init
  --fleet` help, the `fleet` docstring, the scaffolded `env.example` /
  `docker-compose.yml` comments, `README.md`, `CLAUDE.md`, and
  `docs/gateway-fleet.md` still described the fleet as a "2-model" /
  "two-container" / "single-backend" deployment. They now describe the default
  fleet as the generate primary plus co-resident embedding + reranker gears behind
  one gateway, routed by task family (generate / embed / score / rerank), with the
  *generate* fallback as the only opt-in backend. Added a "Task families & gears"
  section + `explain embeddings` / `explain rerank` pointers to `model learn`.

## [0.22.0] - 2026-06-19

### Added

- **Embedding + reranker gears (closes #44).** model-gear now serves two pooling
  gears alongside the chat primary, reachable through the same OpenAI-compatible
  gateway and routed by the request's `model` field:
  - `Qwen/Qwen3-Embedding-0.6B` — `POST /v1/embeddings` (vLLM
    `--runner pooling --convert embed`), native **1024-dim**, MRL-truncatable via
    the `dimensions` param (Matryoshka `--hf-overrides`).
  - `Qwen/Qwen3-Reranker-0.6B` — `POST /v1/rerank` + `/v1/score` (vLLM
    `--runner pooling --convert classify`, served via the
    `Qwen3ForSequenceClassification` `--hf-overrides`).
  - **Catalog:** `SupportedModel` gains `task` (`generate`/`embed`/`score`),
    `dimension`, and `hf_overrides`; both gears surface in `model overview --list`
    and `GET /v1/models/supported`.
  - **Fleet:** `vllm-embed` + `vllm-rerank` services in the fleet compose
    (always-warm, small `--max-model-len`/`--gpu-memory-utilization` so they
    co-reside with the 27B on a single GB10), wired as gateway backends.
  - **Gateway:** task-aware failover — an embed/score request never fails over to
    a generate backend (and vice versa); chat primary↔fallback failover preserved.
  - **CLI:** `model switch --task {generate,embed,score}` for solo serving;
    `model explain embeddings` / `rerank` / `score` document the call shapes;
    per-model docs under `docs/`.
  - **Boundary:** model-gear *serves* the gears only — no vector store, index,
    chunker, or retrieval lands here (guarded by a test); storage + retrieval are
    the consumer's half (eidetic-cli).

### Changed

### Fixed

## [0.21.1] - 2026-06-19

### Fixed

- **markdownlint:** exempt skill prompt templates (`.claude/skills/**/prompts/**`)
  from markdownlint. These are model-facing prompts fed verbatim to a backend
  (first line is `$ARGUMENTS` or a prose instruction), so MD041 (first-line H1)
  and MD032 are inapplicable — a heading would be injected into the prompt.
  `SKILL.md` is still linted; only `prompts/` is exempt. Unblocks the `lint` CI
  job after the `ask-colleague` skill was vendored in.

## [0.21.0] - 2026-06-12

### Changed

- **The fleet is now single-backend by default (Qwen primary only); the Mistral
  fallback is removed.** Live validation showed two ~30B NVFP4 models don't co-fit
  a shared GB10, so the warm dense Mistral-Small-3.2-24B fallback has been dropped
  from the default fleet and the primary restored to its **load-tested solo
  headroom**: `PRIMARY_GPU_MEM_UTIL` `0.40 → 0.6` and `PRIMARY_MAX_MODEL_LEN`
  `32768 → 262144` (full 256K). The `vllm-fallback` service is gone from
  `fleet/docker-compose.yml`, and `FLEET_CONTAINERS` no longer includes it.
- **The gateway makes the fallback optional.** `build_config` now adds a second
  backend **only** when `FALLBACK_URL` or `FALLBACK_SERVED_NAME` is set in env —
  so the default gateway serves the primary alone (no failover target), and a
  two-backend fleet still works for anyone who wires one up. Routing/failover
  primitives are unchanged; `order_backends` returns just the primary when solo.
- **Mistral stays a selectable catalog candidate** (`model overview --list`) and
  the documented opt-in fallback — only its role as the *default* fleet fallback
  is removed. README, `docs/gateway-fleet.md`, and the `model explain
  fleet/gateway` / `model init --help` text are updated to the single-backend
  default (with an "Adding a fallback" guide).

### Fixed

- `docs/gateway-fleet.md` uses `$HOME/.model-gear` instead of the non-portable
  `~/.model-gear`.

## [0.20.1] - 2026-06-12

### Fixed

Qodo review of #41:

- **`model init --fleet --audio` now scaffolds `_readiness.py`** — added
  `fleet/_readiness.py → _readiness.py` to `_compose.AUDIO_TEMPLATES`. The
  Parakeet `Dockerfile.parakeet` `COPY _readiness.py` requires it at the
  deployment-dir root, so a clean audio init previously produced a tree where
  `docker compose build stt` would fail. Covered by `test_init.py`.
- **Parakeet readiness drift guard + simplification** — removed the third
  (inline) copy of the readiness decision from `listen_server.py` (the scaffold
  now guarantees the vendored `_readiness.py` is present), and added a test
  asserting the vendored twin stays behaviourally identical to the canonical
  `model_gear/realtime/_readiness.py`.
- **CUDA readiness probe failures are now logged** — `listen_server.health()`
  emits a `logger.warning` with the exception type/message before returning
  `503`, so operators can distinguish driver-down / OOM / stale-context.
- **`scripts/audio-smoke.py` now exercises `/v1/audio/speech`** (it previously
  claimed both routes but only tested transcriptions) and wires the formerly
  unused `--stt-url` to a direct-Parakeet transcription check.
- **`docs/realtime-pipeline.md`** uses `$HOME/.model-gear` instead of the
  non-portable `~/.model-gear`.

## [0.20.0] - 2026-06-12

### Added

- **`docs/realtime-pipeline.md`** — the previously-missing runbook for the audio
  surface: that model-gear owns the live `:8080` realtime facade, the
  `model init --fleet --audio` / `model fleet up` bring-up, the topology
  (gateway path-routes `/v1/audio/*` → realtime → Parakeet/Magpie), the drift it
  fixed (#39/#40), the cheap readiness probe, and the stale-Parakeet-CUDA restart
  runbook. Resolves a doc referenced from `pyproject.toml`, the audio overlay,
  and the realtime app docstring but never written.
- **`scripts/audio-smoke.py`** — a stdlib-only live smoke test for the audio
  routes: asserts `GET :8080/openapi.json` lists both `/v1/audio/transcriptions`
  and `/v1/audio/speech`, then POSTs an in-memory 16 kHz WAV and asserts
  `200 {text: …}`. Reproduces issue #39's repro to confirm the 500→200 fix.
  Requires a running GPU box (not a CI unit test).
- **`model_gear/realtime/_readiness.py`** — a stdlib-only `evaluate_readiness()`
  helper backing the Parakeet `/v1/health/ready` cheap probe; unit-tested in CI
  without torch/nemo/GPU.

### Fixed

- **Parakeet STT healthcheck now reflects real model readiness (#39).** The
  vendored `templates/fleet/listen_server.py` `/v1/health/ready` returned
  `{"status": "ready"}` unconditionally — process liveness only — so a container
  whose CUDA context had gone stale (`CUDA error: unknown error`, every
  transcription 500ing) still reported Docker "healthy". The probe now reports
  ready **only** when the NeMo model is loaded **and** a trivial CUDA tensor op
  succeeds, returning `503` otherwise (a cheap probe, not a full transcription
  each interval). The pure decision is vendored into the Parakeet build context
  and `COPY`'d into the image so it resolves without the wheel.

## [0.19.0] - 2026-06-09

### Added

- **`scripts/gen-api-key.py`** — generate or rotate the bearer key
  (`CULTURE_VLLM_API_KEY`) that gates the served API. The secret is created with
  the stdlib `secrets` module and **never hardcoded**, so the script is safe in the
  open-source repo; the key only ever lands in the gitignored deployment `.env`
  (written `0o600`, best-effort). Hidden by default (no echo into logs/scrollback);
  `--show` prints it, `--force` rotates an existing key, and `--bytes` (min 16) is
  validated. Resolves the deployment dir like the `model` CLI (`--dir` →
  `$MODEL_GEAR_DIR` → `$HOME/.model-gear`), degrades gracefully on an unreadable or
  non-regular `.env`, and runs from a wheel install (no `model_gear` import).
  Referenced from the README "Expose the API" section.

## [0.18.0] - 2026-06-09

### Added

- **`model tunnel` — expose the local OpenAI-compatible API from anywhere via a
  Cloudflare Tunnel** (#35). Dry-run by default (prints the `cloudflared` command
  and the public `https://<host>/v1` URL); `--apply` starts a standalone
  `cloudflared tunnel run` in the background (logging to `cloudflared.log` in the
  deployment dir), and `--stop --apply` tears it down. The public hostname resolves
  `--hostname` → `$CULTURE_VLLM_PUBLIC_HOSTNAME` → `CULTURE_VLLM_PUBLIC_HOSTNAME` in
  a **gitignored** `.cf-tunnel.env`; the run-token comes from
  `CULTURE_CF_TUNNEL_TOKEN_SHUSHU` (a shushu-sealed secret name, preferred) or
  `CULTURE_CF_TUNNEL_TOKEN` (plaintext fallback). The token is **never placed on the
  process argv** (so it can't leak via `ps` or the log) — cloudflared reads it from
  the `TUNNEL_TOKEN` environment variable, which `shushu` injects (sealed mode) or
  the launcher sets directly (fallback). The resolved hostname and sealed-secret
  name are validated against a conservative charset before they reach the argv (an
  argument-injection guard). `--apply` preflights that `cloudflared` (and `shushu`)
  is on PATH, that no tunnel is already running for the deployment, and that the
  local server answers `/health`; `--stop` signals the recorded process *group* and
  confirms exit (SIGTERM → SIGKILL) before clearing a PID-reuse-safe pidfile (the
  recorded pid is identity-checked against `/proc` so a reused pid can't be killed).
  No hostname, token, or backend checkpoint id is committed. The Cloudflare side
  (tunnel + ingress + DNS) is provisioned once by `cultureflare remote-login
  --no-access`.
- **Optional bearer auth on the served API** via `CULTURE_VLLM_API_KEY`, wired into
  the single-model `docker-compose.yml` as `VLLM_API_KEY=${CULTURE_VLLM_API_KEY:-}`.
  Empty (default) leaves local dev open; set it and vLLM requires `Authorization:
  Bearer` — the gate for any public exposure. Documented in `env.example` alongside
  a note that `VLLM_SERVED_NAME` can be a generic alias to keep the checkpoint name
  out of the public `/v1/models`.
- **`cf-tunnel.env.example`** scaffolded by `model init` (single + fleet), a
  placeholder-only template the owner copies to the gitignored `.cf-tunnel.env`.
- README "Expose the API from anywhere (Cloudflare Tunnel)" section and a
  `model explain tunnel` catalog entry.

## [0.17.0] - 2026-06-03

### Changed

- **Served context raised 128K → full 256K (native) for the MTP primary on DGX
  Spark.** The `spark` machine profile's `max_model_len` default is now `262144`
  (was `131072`), with matching changes to the single-model `env.example` /
  `docker-compose.yml` defaults and the `model switch --help` / `model explain`
  text. Load-tested 2026-06-03 on the shared GB10 (util 0.6, `--max-num-seqs 2`,
  KV-FP8, MTP n=3): boots clean (CUDA-graph capture, PIECEWISE, **0.71 GiB** in 2 s
  — **no OOM**), **17.8 tok/s** decode, **74.0 %** MTP draft acceptance, both
  `model assess` probes `finish=stop`, tool-calling probe passes, and **71,601 MiB
  (~70 GiB)** resident — the *same* footprint as 32K/128K, because
  `--gpu-memory-utilization` fixes the KV-pool reservation (only the addressable
  context grows). vLLM reports **5.29× max concurrency at a full 256K request**,
  well above the `--max-num-seqs 2` decode cap, so **there is no practical
  concurrency cost** versus the 128K default. `model switch --max-model-len <N>`
  still overrides per deployment, and util stays a conservative `0.6` (shared box).
  See `docs/qwen3.6-27b-text-nvfp4-mtp.md` (new 256K benchmark) and
  `docs/tuning-profiles.md`.
- **Catalog `context` string updated.** The MTP primary now reads
  `"256K native (served at full 256K on the shared GB10)"`.
- **Scope — deliberately left at the old contexts:** fleet templates stay at 32K
  (co-residence with the 24B fallback is a different, still-unvalidated memory
  regime; `fleet/env.example` notes this), and the `thor` / `generic` machine
  profiles stay at 32K (unmeasured estimates) with `blackwell` at 64K. The
  `model switch` native-ceiling clamp (added in 0.16.0) still pins 32K-native
  candidates (`nvidia/Qwen3-32B-NVFP4`, `mmangkad/Qwen3.6-35B-A3B-NVFP4`) down to
  their own ceilings under the new 256K spark default.

### Fixed

- **`model switch` warns when an uncatalogued model would inherit an unclamped
  machine context default.** The native-ceiling clamp only protects catalogued
  models; an uncatalogued model ID (which `switch` supports) inherits the machine
  default (now spark's 262144) and would boot-fail if the checkpoint's native
  context is smaller. `switch` now emits a clear warning pointing at
  `--max-model-len` / cataloguing, rather than silently applying the high default
  (no silent clamp — an uncatalogued ceiling is unknown, so guessing one is wrong
  both ways). Addresses a Qodo reliability finding on #34.

## [0.16.0] - 2026-06-03

### Changed

- **Served context raised 32K → 128K for the MTP primary on DGX Spark.** The
  `spark` machine profile's `max_model_len` default is now `131072` (was `32768`),
  with matching changes in the single-model `env.example` /`docker-compose.yml`
  defaults. Load-tested 2026-06-03 on the shared GB10 (util 0.6, `--max-num-seqs 2`,
  KV-FP8, MTP n=3): boots clean (no CUDA-graph-capture OOM), **18.3 tok/s** decode,
  **73.3 %** MTP draft acceptance, both `model assess` probes `finish=stop`, and
  **71,963 MiB (~70 GiB)** resident — the *same* footprint as 32K, because
  `--gpu-memory-utilization` fixes the KV-pool reservation (the pool holds **9.6×**
  a full 128K request). `model switch --max-model-len <N>` still overrides per
  deployment, and util stays a conservative `0.6` (the box is shared). See
  `docs/qwen3.6-27b-text-nvfp4-mtp.md` (new 128K benchmark) and
  `docs/tuning-profiles.md`.
- **Catalog `context` strings clarified.** The MTP primary now reads
  `"256K native (served at 128K on the shared GB10)"`; the non-served candidate /
  fallback entries (`mmangkad/Qwen3.6-27B-NVFP4`, the Mistral fallback) drop the
  stale per-model "capped to 32K" note and state native context only.
- **Scope — deliberately left at the old contexts:** fleet templates stay at 32K
  (the fleet runs the primary co-resident with a 24B fallback at lower util — a
  different memory regime the single-model 128K test does not validate;
  `fleet/env.example` notes this), and the `thor` / `generic` machine profiles
  stay at 32K (unmeasured estimates) with `blackwell` at 64K.

### Fixed

- **`model switch` clamps the machine context default to a model's native ceiling.**
  Raising spark's `max_model_len` default to `131072` made it apply to *every* model
  switched to on spark — including the 32K-native catalog candidates
  (`nvidia/Qwen3-32B-NVFP4`, `mmangkad/Qwen3.6-35B-A3B-NVFP4`), where vLLM refuses a
  `--max-model-len` above the checkpoint's native limit (no YaRN) and the container
  fails to boot. `SupportedModel` now carries a numeric `native_max_model_len`, and
  `model switch` clamps the resolved context *down* to it when no explicit
  `--max-model-len` is given (an explicit value still wins, for opted-in YaRN
  configs). Fixes a Qodo correctness finding on #33.

## [0.15.0] - 2026-05-31

### Changed

- **Fleet default primary → `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`** (the MTP
  build), replacing `mmangkad/Qwen3.6-27B-NVFP4` (issue #26 follow-up). The
  tool-calling gate that kept it a candidate is now **closed**: served through the
  production compose it emits a valid `qwen3_coder` tool call, completes a full
  tool round-trip, keeps its reasoning trace, and runs MTP spec-decode at **78.6%
  draft acceptance with tool calling on** — ~2.4× single-stream decode (8 → ~19
  tok/s), ~71 GB footprint, both `model assess` probes `finish=stop`. Promoted
  across the catalog (`role_hint`), the gateway default (`_DEFAULT_PRIMARY`),
  `whoami`, both template `env.example`/`docker-compose.yml` files, and
  `culture.yaml`.
- **The MTP serve flags are now baked into the compose templates** (single-model +
  fleet `vllm-primary`): `--speculative-config`, `--trust-remote-code`,
  `--language-model-only`, the `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` override,
  and `--max-num-seqs=2`. A fresh `model init && model serve` of the default now
  works out of the box. Quantization default is `modelopt`.
- **`model switch` notices inverted.** Because the template ships the MTP primary's
  flags, switching to a **non-MTP** model now prints "REMOVE these 4 `command:`
  lines" (was "add" for the MTP candidate); the MoE `--moe-backend` add-notice is
  unchanged. Switching to the MTP primary force-caps `--max-num-seqs` to 2.
- **`mmangkad/Qwen3.6-27B-NVFP4` archived to a candidate** — retained as the MTP
  primary's tokenizer source and the only vision-capable 27B in the catalog.

### Fixed

- **`model switch --apply` no longer takes a healthy deployment down when a manual
  compose edit is required** (Qodo review). Switching to a non-MTP model (the
  template ships the MTP primary's incompatible flags) now writes `.env` and
  **stops before the restart**, printing the lines to remove; `--force` overrides
  to recreate the container anyway.
- **MTP compose flags are a single source of truth** (`catalog.mtp_compose_command_items()`) —
  consumed by both `model switch`'s removal notice and guarded against drift from
  the packaged templates by a new test (Qodo review).
- **Security guidance for the now-default `--trust-remote-code`** added to both
  compose templates and `env.example`: HF_TOKEN is only needed for gated repos
  (defaults are public) — leave it empty or use a minimal-scope read-only token, and
  pin trusted revisions (Qodo review). Tracking the upstream tokenizer fix that would
  let us drop the override in #29.

## [0.14.0] - 2026-05-31

### Added

- **MTP (Multi-Token Prediction) candidate for the 27B** (issue #26). New catalog
  entry `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` — a text-only re-export of the
  27B primary with its **MTP draft head restored in bf16** so vLLM speculative
  decoding actually works. The lesson from the 35B MoE applied: the baseline NVFP4
  export drops the MTP head (~0 % draft acceptance), and a newer vLLM isn't
  installable on the aarch64 GB10 — so the fix is *a checkpoint that ships the MTP
  weights*, not a newer engine. Carries a catalog `speculative_config`
  (`{"method":"qwen3_5_mtp","num_speculative_tokens":3}`); quantization is
  `modelopt`. **Load-tested on the DGX Spark (GB10) 2026-05-31: 19.1 tok/s decode
  (~2.4× the baseline 27B's ~8 tok/s) at 72 % MTP draft acceptance on vLLM
  0.19.0+nv26.04** — the open risk (does the stock image accept `qwen3_5_mtp`?) is
  cleared (it resolves the `Qwen3_5MTP` draft head). One tokenizer override is
  required: the checkpoint declares the newer `TokenizersBackend` class (absent from
  nv26.04), so serve with `--tokenizer=mmangkad/Qwen3.6-27B-NVFP4` (the cached
  sibling, same vocab); `model switch` prints it.
  - New per-model doc `docs/qwen3.6-27b-text-nvfp4-mtp.md` with the serve recipe,
    the live benchmark table (decode tok/s + acceptance vs the baseline), and the
    caveats (`--max-num-seqs 2` or it silently OOMs; the tokenizer override).

### Changed

- **`model switch` surfaces MTP serve-extras, not just MoE.** `_moe_notice` →
  `_serve_notices` (now a list): a model with a catalog `speculative_config` prints
  the exact `--speculative-config` / `--trust-remote-code` / `--language-model-only`
  compose edits (+ the `VLLM_MAX_NUM_SEQS=2` reminder), the same hand-edit pattern
  as `--moe-backend`. The `--json` dry-run replaces the `moe_notice` key with a
  `compose_edits` list. `env.example` + the `explain` catalog prose updated to match.

### Fixed

## [0.13.0] - 2026-05-31

### Added

- **Workload `purpose` + machine tuning profiles.** `model switch` now resolves
  the serve config from three layers — a **machine** profile (`--machine`,
  default auto-detected from `nvidia-smi` + hostname: GPU-memory fraction,
  context, attention backend), a **workload** profile (`--purpose`, default
  `balanced`: the batching knobs and the shape `model benchmark` exercises), and
  the model's catalog entry — with explicit `--max-model-len` / `--gpu-mem-util`
  flags overriding the machine defaults.
  - **New `model_gear/profiles.py`** (pure data module, like `catalog.py`):
    `WorkloadProfile` (`balanced` ≈1K/1K, `prompt-heavy` ≈8K/1K, `decode-heavy`
    ≈1K/8K) and `MachineProfile` (`spark` load-tested, `thor`/`blackwell`/`generic`
    configured), guarded by `tests/test_profiles.py`.
  - **Richer single-model template** — the serve command now passes
    `--attention-backend`, `--max-num-seqs`, `--max-num-batched-tokens` (env-driven),
    plus static `--enable-chunked-prefill` / `--async-scheduling`. New `.env` keys:
    `VLLM_PURPOSE`, `VLLM_MACHINE`, `VLLM_ATTENTION_BACKEND`, `VLLM_MAX_NUM_SEQS`,
    `VLLM_MAX_NUM_BATCHED_TOKENS`.
  - **Per-model MoE serve extras** — the catalog gains `moe_backend` /
    `speculative_config` (set only on the `Qwen3.6-35B-A3B` MoE candidate).
    `model switch` to the MoE prints them as a documented compose edit (they break
    the dense/hybrid models and can't be defaulted in the shared template).
  - **`model benchmark` is tied to the config** — its workload shape defaults to
    the configured `VLLM_PURPOSE` (overridable with `--purpose` / `--input-len` /
    `--output-len`).
  - `model whoami` / `model overview` surface the active `gear` (purpose/machine);
    `model explain tuning` documents the layering; `docs/tuning-profiles.md` is new.
  - Credit: the serve tuning and the three workload shapes follow **shahizat**'s
    cross-machine NVFP4 benchmark (NVIDIA Developer Forums) — see the README
    Acknowledgements and `docs/tuning-profiles.md`.
  - **Live-replicated on the shared DGX Spark (2026-05-31)** rather than trusting
    the post: with the new flags the **35B MoE candidate loads solo** (util 0.70,
    marlin) and runs **single-stream decode ~35 tok/s vs the 27B's ~7.8 — ~4.6×
    faster** (the MoE's ~3B-active advantage). Numbers + method in
    `docs/tuning-profiles.md` and `docs/qwen3.6-35b-a3b-nvfp4.md`.

### Changed

- `model switch` `--max-model-len` / `--gpu-mem-util` now default to the machine
  profile (was a fixed 32768 / 0.6); pass them explicitly to override.
- `model benchmark` replaces `--decode-tokens` with purpose-driven
  `--input-len` / `--output-len`.
- **Catalog: dropped the MTP `speculative_config` from the `mmangkad/Qwen3.6-35B-A3B-NVFP4`
  entry** (kept `--moe-backend=marlin`). Live testing showed shahizat's MTP draft
  fails to load on the `mmangkad/` copy (`qwen3_5_mtp.py` weight-shape mismatch on
  vLLM nv26.04) — it is tied to his `nvidia/` checkpoint. `model switch` no longer
  prints a recipe that wouldn't load.

### Fixed

## [0.12.0] - 2026-05-30

### Added

- **Audio I/O behind the gateway (STT + TTS) — issue #18, part 1 of 3.** model-gear
  now serves OpenAI-compatible `POST /v1/audio/transcriptions` and
  `POST /v1/audio/speech` on the same host port as the text API, fronted by the same
  stdlib gateway. The audio backends are the *same models* the standalone realtime-api
  stack ran — **NVIDIA Parakeet STT** + **Magpie TTS NIM** — consolidated into the
  fleet (no separate compose project; the realtime bridge's LLM is the fleet gateway
  itself, so there is no extra vLLM container).
  - **New `[realtime]` extra + `model_gear.realtime` package** (vendored from the
    `realtime-api` sibling, cite-don't-import): a FastAPI bridge that exposes the
    OpenAI audio surface (`/v1/audio/speech` adapts Magpie's proprietary
    `/v1/audio/synthesize`; `/v1/audio/transcriptions` forwards to Parakeet). The base
    wheel and the gateway stay stdlib-only — torch/fastapi never leak into them.
  - **Gateway audio routing** — `/v1/audio/*` is path-routed to the audio backend
    (`AUDIO_URL`) with no model rewrite and no failover; binary responses relayed
    **streamed** (chunked) so a large TTS body never buffers whole in the gateway.
    Unset `AUDIO_URL` (a text-only fleet) → those paths 404, unchanged.
  - **`model init --fleet --audio`** scaffolds the audio overlay
    (`docker-compose.audio.yml` + `Dockerfile.realtime` + a vendored
    `Dockerfile.parakeet`/`listen_server.py`) and appends the audio keys to `.env`.
    `model fleet up`/`down`/`status` auto-include the overlay when present.
  - **Co-residence caveat:** the audio services share the GPU with the LLM fleet — the
    overlay is opt-in so text-only boxes keep their GPU budget. See the per-model docs
    (PR3) for live numbers.
  - **The realtime WebSocket (`/v1/realtime`) and the `model overview`/`doctor`/`explain`
    surface land in the follow-up PRs (parts 2 and 3).**

### Changed

### Fixed

- **Audio review hardening (PR #24 review).**
  - **Gateway no longer buffers whole audio bodies** — `/v1/audio/*` responses are
    relayed chunked instead of `read_all()`'d into memory, so one large TTS WAV can't
    OOM the fleet's single front door.
  - **`TTS_CONCURRENCY` / `TTS_SPEED` clamped to ≥ 1** — `TTS_CONCURRENCY=0` previously
    seeded an `asyncio.Semaphore(0)` that hung every TTS request; a 0/negative speed
    emitted nonsensical `rate="0%"` SSML.
  - **`/v1/audio/speech` `speed` clamped to OpenAI's 0.25–4.0 range** before the Magpie
    percentage conversion, so out-of-range values no longer reach the backend as
    `rate="{huge|negative}%"` and 502.
  - **SonarCloud config** — coverage exclusions now mirror `coverage.run` `omit` (the
    `[realtime]`-extra modules can't be unit-imported offline), and the deployment
    *scaffolds* under `model_gear/templates/**` are excluded from analysis (container
    Dockerfiles + the vendored Parakeet server aren't package runtime). Added unit
    tests for `realtime.protocol`, the settings clamps, the speed clamp, and the
    streamed audio relay.

## [0.11.1] - 2026-05-30

### Added

- **`model learn --json` now includes a `models` object** (`supported_catalog` /
  `loaded_now`) — a machine-readable version of the catalog-vs-loaded explainer for
  agent consumers. (Additive field; the only observable behavior change in this
  release.)

### Changed

- **Documented "supported catalog vs. loaded now" consistently** across the README,
  `docs/gateway-fleet.md` (new "Supported catalog vs. warm backends" subsection),
  the per-model docs, and the CLI teaching surfaces (`model learn`,
  `model explain models`/`overview`/`status`/`whoami`/root, and the
  `overview`/`status`/`whoami`/`fleet status` help strings). The distinction:
  `model overview --list` / `GET /v1/models/supported` = the gears you *can* switch
  to (tagged `load-tested`/`configured`, static); the live `GET /v1/models` (which
  `model fleet status` queries) = what's actually *loaded* now. `model status` /
  `model whoami` report the *configured* served model (from `.env`) + health — not
  a live `/v1/models` query. Docs + help text (no serving/runtime behavior change).

## [0.11.0] - 2026-05-30

### Added

- **`RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4` support** — added to the
  supported-model catalog (`model overview --list`, `GET /v1/models/supported`)
  with a per-model doc, [`docs/mistral-small-3.2-24b-nvfp4.md`](docs/mistral-small-3.2-24b-nvfp4.md).
  Load-tested on the DGX Spark (GB10): ~15 GiB weights, **~14.9 tok/s** decode,
  prefill 2,009 tok in 1.49 s, tool calling ✅.
- **`mistral` tool-call parser inference** — `model_gear.runtime._parser` now maps
  Mistral-family ids (incl. the `mistralai/` org) to the `mistral` parser; `model
  switch` auto-selects it.
- **`model switch --quantization`** — the served `--quantization` is now set per
  model (read from the catalog for a known model, e.g. `compressed-tensors` for the
  RedHatAI NVFP4 Mistral vs `modelopt_fp4` for the nvidia/mmangkad checkpoints);
  `--quantization` overrides it. The single-model compose reads `VLLM_QUANTIZATION`.

### Changed

- **Fleet default fallback is now the dense Mistral-Small-3.2-24B**, replacing the
  `mmangkad/Qwen3.6-35B-A3B-NVFP4` MoE, which never loaded on the GB10 (OOM
  co-resident, stall solo — no benchmark obtained). Mistral is dense, loads
  reliably, and is smaller (~15 GiB weights). The fleet compose serves it with the
  **mistral tokenizer + images limited to 0** (required for tool-call parsing on
  the nv26.04 build; the HF tokenizer leaks `[TOOL_CALLS]` markup, and the mistral
  tokenizer alone crashes the Pixtral profiler) and **no** `--reasoning-parser`
  (instruct model). The 35B MoE is demoted to a catalogue candidate.
- `model_gear.gateway._config._DEFAULT_FALLBACK`, the fleet `docker-compose.yml` /
  `env.example` `FALLBACK_*` defaults, `docs/gateway-fleet.md`, and `README.md`
  updated for the new fallback.

### Fixed

## [0.10.1] - 2026-05-30

### Changed

- **Fleet default GPU-mem utilisations rebalanced `0.55`/`0.30` → `0.40`/`0.35`.**
  Live validation on a DGX Spark (GB10) showed `0.55`/`0.30` OOM-crash-loops the
  fallback: the 27B primary alone takes ~75 GiB at util 0.6, and `--gpu-memory-utilization`
  is fraction-of-total *per process* (the two backends don't coordinate). The new
  values are a dedicated-box estimate; the templates and docs now state plainly
  that co-residence of two ~30B models needs a dedicated box.

### Fixed

- **Docs corrected against live findings (2026-05-30):** `docs/gateway-fleet.md`
  gains a "Live validation findings" section (27B warm-up ~7 min, ~75 GiB footprint,
  8.0 tok/s decode; co-residence not viable on a shared GB10).
  `docs/qwen3.6-35b-a3b-nvfp4.md` updated from "not yet load-tested" to the actual
  result — the MoE fallback does **not** load reliably on this box (OOM co-resident;
  crash/stall even solo). `docs/qwen3.6-27b-nvfp4.md` reframed as the fleet default
  primary (was "candidate") with the warm-up measurement and a corrected
  recommendation.

## [0.10.0] - 2026-05-30

### Added

- **`GET /v1/models/supported` gateway endpoint — the "change gears" catalog.**
  Alongside the OpenAI-standard `/v1/models` (which lists only the two *loaded*
  backends), the gateway now serves the full catalog of supported models a client
  can change gears to, each flagged `loaded` (a backend serves it now) and
  `default` (the gateway routes unknown/missing names there). Non-OpenAI shape
  (`"object": "model-gear.supported_models"`) so `/v1/models` stays standard for
  existing clients. Pure `supported_models_payload()` in `gateway/_routing.py`.
- **New packaged catalog `model_gear/catalog.py`** — a dependency-free
  `SUPPORTED_MODELS` tuple (the 27B primary, the 32B dense candidate, the 35B-A3B
  MoE fallback) that is the single source of truth for both the gateway (which
  runs from a wheel and can't read `docs/`) and the CLI. `model overview --list`
  is now catalog-backed, so it is populated even in a wheel install.

### Changed

- **Fleet (and single-model) default primary → `mmangkad/Qwen3.6-27B-NVFP4`.**
  The scaffolded default served model is now the Qwen3.6 27B (hybrid
  Mamba/linear-attn + ViT, 256K native context) with `--tool-call-parser=qwen3_coder`
  — matching what runs on the DGX Spark and convertible's parent model. The dense
  `nvidia/Qwen3-32B-NVFP4` remains a supported candidate (`PRIMARY_MODEL` /
  `model switch`). Recomputed co-resident GPU memory: `PRIMARY_GPU_MEM_UTIL=0.55`
  and `FALLBACK_GPU_MEM_UTIL=0.30` (the 27B is heavier than the 32B). Updated the
  fleet + single-model templates, `gateway/_config.py`, `whoami` default,
  `culture.yaml` / `AGENTS.md` / `CLAUDE.md` (served-model coherence chain), and
  the per-model + gateway-fleet docs.

### Fixed

## [0.9.0] - 2026-05-28

### Added

- **Fallback model + single front OpenAI gateway ("fleet").** A new
  scaffold-based deployment runs **two always-warm vLLM backends behind one
  stdlib gateway** that model-gear manages as three containers
  (`model-gear-gateway`, `model-gear-vllm-primary`, `model-gear-vllm-fallback`).
  The gateway routes each request by its `model` field, defaults an
  unknown/missing name to the primary, and fails over to the other backend when
  the chosen one refuses the connection or returns a 5xx **before** the response
  body (4xx is returned verbatim; no mid-stream retry). SSE streams are relayed
  chunk-by-chunk. Default fallback: the MoE `mmangkad/Qwen3.6-35B-A3B-NVFP4`.
- **New gateway package `model_gear/gateway/`** — a pure-stdlib
  (`http.server` + `http.client`, no runtime deps) reverse proxy: `_routing.py`
  (pure name/alias/default routing + failover ordering), `_config.py` (env →
  routing table + server config), `server.py` (the `handle_post` failover seam,
  upstream client, and `ThreadingHTTPServer` handler), run as
  `python -m model_gear.gateway`.
- **`model init --fleet`** scaffolds the fleet templates
  (`docker-compose.yml` + `.env` + `Dockerfile.gateway`) and pins
  `MODEL_GEAR_VERSION` to the running release; **`model fleet up | down |
  status`** drives the deployment (`up`/`down` dry-run by default, `--apply` to
  commit; `status` is read-only and reports all three containers + the gateway
  `/health` + `/v1/models`).
- **Docs:** `docs/gateway-fleet.md` (topology, routing/failover, memory,
  verbs), `docs/qwen3.6-35b-a3b-nvfp4.md` (the MoE fallback), a README "fleet"
  section, and `model explain fleet` / `model explain gateway` entries.

### Changed

- `model_gear/runtime/_compose.py` gained a template registry
  (`SINGLE_TEMPLATES` / `FLEET_TEMPLATES`), a `templates=` argument on
  `scaffold_plan` / `write_scaffold` (single-model stays the default — existing
  callers unchanged), a `compose_up_build` helper, and `FLEET_CONTAINERS`.
- The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` /
  `VLLM_TOOL_CALL_PARSER` (= the primary) so the read-only single-model verbs
  (`status` / `whoami` / `doctor`) stay coherent on a fleet deployment.
  `model switch` remains single-model only.

### Fixed

## [0.8.1] - 2026-05-27

### Changed

- **SonarCloud cleanup (no behavior change).** Split `cmd_switch` into
  `_select_parser` / `_emit_dry_run` / `_apply_switch` helpers to bring its
  cognitive complexity under the gate, and hoisted the repeated `"(unset)"`
  literal in `model status` into a `_UNSET` constant.

## [0.8.0] - 2026-05-27

### Added

- **Per-model tool-call parser auto-selection.** New `model_gear/runtime/_parser.py`
  `infer_parser()` maps a model name to its parser (`qwen3_coder` for
  Qwen3-Coder / Qwen3.6, `hermes` for Qwen3 dense, unknown → leave untouched).
  `model switch` now picks the right parser automatically so tool calling keeps
  working across a switch without the caller remembering it; `--tool-call-parser`
  still overrides ([issue #13](https://github.com/agentculture/model-gear/issues/13)).
- **Post-switch / post-start tool-calling probe.** `model switch --apply` and
  `model serve --apply` now probe `tool_choice:"auto"` once the container is
  healthy and report PASS/FAIL (with the called tool names) — reusing the
  existing `assess` probe. `--no-probe` skips it; the probe never aborts the
  command (unreachable / HTTP 400 degrade to a FAIL result).
- **`model status` reports the active `tool_call_parser`** (`VLLM_TOOL_CALL_PARSER`),
  so "which gear am I in" is complete without `docker inspect`.

### Changed

- **`lepenseur` is retired; the deployed agent is now `model-gear`.** The tool and
  the deployed agent share one identity. Updated `culture.yaml` (`suffix: model-gear`),
  the `AGENTS.md` system prompt, `model whoami` / `learn` / `explain` output, the
  posting nick (`.claude/skills.local.yaml.example`), the compose/`.env` templates,
  `README.md`, and `CLAUDE.md` (the former "two identities" section now describes
  one).

### Fixed

## [0.7.0] - 2026-05-27

### Added

- **OpenAI tool/function calling** on the served vLLM model. The packaged compose
  template (`model_gear/templates/docker-compose.yml`) now serves with
  `--enable-auto-tool-choice` and `--tool-call-parser=${VLLM_TOOL_CALL_PARSER:-hermes}`,
  so `tool_choice:"auto"` requests return a `tool_calls` array instead of HTTP
  400. Additive — plain chat/reasoning is unaffected, no extra GPU/memory cost.
  Unblocks coder-agent harnesses that drive the model entirely through tool calls
  ([issue #9](https://github.com/agentculture/model-gear/issues/9)).
- **`VLLM_TOOL_CALL_PARSER`** env var (default `hermes`) + **`model switch
  --tool-call-parser`** — the parser is per-model: `hermes` fits Qwen3 dense
  (e.g. `Qwen3-32B`), while Qwen3-Coder / Qwen3.6 checkpoints emit the XML
  function format and need `qwen3_coder`. `switch` writes the var only when the
  flag is given, so retuning a model never clobbers its parser.
- **`model assess --tools`** — an opt-in tool-calling probe that verifies a
  `tool_choice:"auto"` request returns a `tool_calls` array naming a `finish`
  function. Degrades gracefully (a FAIL row, no abort) against a server that
  lacks the flags.

### Changed

### Fixed

## [0.6.0] - 2026-05-27

### Added

- **devague workflow trio** vendored under `.claude/skills/` (cite-don't-import):
  `think` (idea→spec), `spec-to-plan` (spec→plan), and `assign-to-workforce`
  (plan→parallel implementation) — the operator chain for the deterministic
  `devague` CLI. Authored in `agentculture/devague`, vendored via guildmaster;
  each carries `type: command` (load-bearing on the culture/agex backend, where
  a `SKILL.md` without `type:` is silently skipped). They drive the `devague`
  CLI at runtime (`uv tool install devague`), resolved portably by the wrappers.
- **`docs/skill-sources.md`** — provenance ledger recording the citation path
  and authoring origin of every vendored skill (the trio plus the six
  steward-sourced skills).

## [0.5.0] - 2026-05-27

Redesigned the repo around **running, assessing, and switching the local vLLM
model**. The model-ops logic that lived in the `model-runner` *skill* is now a
first-class CLI. lepenseur is still the deployed agent that consumes the served
model; model-gear is the tool that runs it.

### Added

- **Model-ops verbs** on the `model` CLI: `switch <model>`, `serve` (alias
  `start`) / `stop`, `status`, `assess` (correctness probes), `benchmark`
  (decode throughput + prefill), and `init` (scaffold a deployment dir). Write
  verbs (`switch`/`serve`/`stop`/`init`) are **dry-run by default** and require
  `--apply` (mutation-safety rule).
- **Scaffold-based deployment.** `docker-compose.yml` + `env.example` ship as
  packaged templates under `model_gear/templates/`; `model init` materialises
  them into `~/.model-gear` (default), a `TARGET`, or the local folder. Every
  model-ops verb resolves the deployment dir via `--compose-dir` →
  `$MODEL_GEAR_DIR` → `~/.model-gear`.
- Ported runtime modules (`model_gear/runtime/` + `model_gear/assess.py`),
  stdlib-only (`urllib`, fixed-argv `subprocess`), with full unit tests.
- `model overview` now folds in the currently-served model and the
  candidate-model list, filterable with `--current` / `--list`.

### Changed

- **PyPI distribution renamed `lepenseur` → `model-gear`; binary `lepenseur` →
  `model`; Python package `lepenseur` → `model_gear`.** Error class
  `LepenseurError` → `ModelGearError`. The `lepenseur` console script is removed.
- Agent-first verbs reframed for the tool: `whoami` reports tool/machine/served
  model/container health/agent; `learn` teaches the model-ops surface; `explain`
  catalog rewritten (`switch`/`assess`/`backend`/`models`/…).
- `doctor` is now **real** — checks docker availability, deployment scaffold,
  `.env` ↔ `culture.yaml` coherence, and `/health` reachability (a down model is
  a warning, not a failure).
- The `model-runner` skill is now a thin shim that `exec`s `model`; its
  `_assess.py` was removed (the logic lives in `model_gear/assess.py`).
- `AGENTS.md` / `culture.yaml` clarified: they describe the deployed `lepenseur`
  agent, not the repo. README + CLAUDE.md reoriented around model-gear.

### Fixed

- **BREAKING:** the vLLM container is renamed `lepenseur-vllm` → `model-gear-vllm`.
  A box running the old container must `docker compose down` under the old name,
  then `model init --apply` + `model serve --apply`.

## [0.4.0] - 2026-05-27

### Added

- `model-runner` skill (local, not vendored): `switch` the local vLLM runtime
  model and `assess`/benchmark it (stdlib `_assess.py` for correctness +
  throughput, host-side facts via the wrapper). Drives this repo's compose +
  `.env`; documented in CLAUDE.md and README. Mutating verbs (`switch`, `down`)
  are dry-run by default and require `--apply` (CLAUDE.md mutation-safety rule);
  `--port` defaults to `.env`'s `VLLM_PORT` (then 8000).

### Changed

- `docs/qwen3.6-27b-nvfp4.md`: filled with the live load-test (DGX Spark/GB10,
  2026-05-27). `mmangkad/Qwen3.6-27B-NVFP4` loads and serves under our vLLM image
  (no `--trust-remote-code`); ~7.9–8.0 tok/s decode, ~70 GB reserved, 29 GB
  weights. It is a hybrid Mamba/linear-attention vision-language model and is
  slower on decode than the 32B here — recommendation: **keep the 32B**. All
  pre-flight caveats (SGLang-only, multimodal, ModelOpt rc) validated/resolved.

## [0.3.0] - 2026-05-27

### Added

- `docs/qwen3-32b-nvfp4.md`: per-model doc for the current runtime model, with a
  live test on DGX Spark (GB10) — `nvcr.io/nvidia/vllm:26.04-py3` (engine
  `0.19.0+...nv26.04`), ~9.7 tok/s decode (batch=1), ~2,800 tok/s prefill, ~72 GB
  reserved at `gpu-memory-utilization=0.6`, correctness verified.
- `docs/qwen3.6-27b-nvfp4.md`: per-model doc for candidate
  `mmangkad/Qwen3.6-27B-NVFP4`. Its `Qwen3_5ForConditionalGeneration` arch is
  registered in the current vLLM image (so the same compose can serve it); live
  load-test/benchmark tracked by issue #6.
- README "Per-model notes" linking both docs.

### Fixed

- `docker-compose.yml`: corrected the `--reasoning-parser=qwen3` comment — on the
  nv26.04 build the `<think>` trace is returned in the `reasoning` field, not
  `reasoning_content`.

## [0.2.0] - 2026-05-27

### Added

- `docker-compose.yml` + `.env.example`: a local vLLM server (NGC
  `nvcr.io/nvidia/vllm` image) that serves the runtime model as an
  OpenAI-compatible API on `:8000` for the `acp` backend, tuned for DGX Spark
  (GB10 Blackwell, 128 GB unified memory).
- README "Running the model locally (vLLM)" section.

### Changed

- Switched lepenseur's runtime model from
  `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` to `nvidia/Qwen3-32B-NVFP4`
  across `culture.yaml`, `AGENTS.md`, `lepenseur/explain/catalog.py`, `README.md`,
  and `CLAUDE.md` (32B dense NVFP4 reasoning model with a thinking mode).

## [0.1.0] - 2026-05-22

### Added

- Initial CLI/PyPI sibling scaffold (copied and adapted from the `lecodeur`
  twin): top-level `lepenseur` package with the `lepenseur` console script.
- Read-only verbs: `whoami`, `learn`, `explain`, `overview`, and a `cli`
  noun with `cli overview`.
- `doctor` verb shipped as a rubric-shaped stub; real self-diagnosis semantics
  for a thinking ("non-doer") agent are deferred to a follow-up.
- Runtime identity files: `AGENTS.md` and `culture.yaml` (acp backend,
  `vllm-local/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`).
- CI: `tests.yml` (test + lint + `afi cli doctor . --strict` gate +
  version-check) and `publish.yml` (PyPI/TestPyPI via Trusted Publishing).
- Six vendored skills under `.claude/skills/` (cicd, communicate, version-bump,
  run-tests, sonarclaude, doc-test-alignment), provenance: steward.
