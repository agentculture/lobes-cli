# Fleet: the cortex + senses roles behind one OpenAI gateway

The **fleet** runs the always-warm Qwen 27B generate primary — the Colleague
**`cortex`** role — and the Gemma 4 12B multimodal gear — the **`senses`**
role — as a **default-on duo**, plus two tiny co-resident **embedding**
(`embedder`) and **reranker** (`reranker`) gears, behind a single stdlib
OpenAI-compatible gateway, managed by lobes as Docker containers. Together with
the opt-in audio overlay's `stt`/`tts` roles, these are the SIX first-class
Colleague-facing roles (issue #81) — see
[`docs/colleague-stack.md`](colleague-stack.md) for the full role contract
(`lobes capabilities`, `GET /capabilities`, `lobes up <role>`, `lobes
measure`). This doc covers the fleet's Docker topology, tuning, and memory
budget; the role contract lives in that sibling doc.

The fleet is an alternative to the bare single-model deployment — scaffold it
with `lobes init --fleet` (the single-model `lobes init` is unchanged and
remains the default). The gateway routes by **task family** (generate / embed
/ score / rerank) and by **capability-tier alias** (`main` / `minor` /
`multimodal`, with back-compat aliases `hard` / `cheap` / `normal`, and the
Colleague-role aliases `cortex` / `senses` layered on top of `main` /
`multimodal`); the 4B `minor` and the legacy 14B are opt-in (see
"Generate-lane tier aliases").

## Why

The single-model deployment serves one model on `:8000` and `lobes switch` swaps
it (freeing the prior model). The fleet instead puts a stable OpenAI endpoint in
front of the primary, so:

- existing clients (the acp `vllm-local` provider, `curl`, …) point at `:8000`
  and keep working — an unknown/missing `model` defaults to the primary;
- the gateway can route additional models by name and fail over **if** a second
  backend is wired up;
- the same front fans `/v1/audio/*` out to the audio overlay (`--audio`).

On the DGX Spark (GB10, 128 GB unified memory) the primary (`cortex`) — a
hybrid-Mamba **27B** — now serves its **full 128K native context at util
0.30**, co-resident with the default-on Gemma 4 12B multimodal gear
(`senses`), which is trimmed to **32K at util 0.14**, as the fleet's
"always-on duo" (retuned 2026-07-02 — see "Memory" below). An earlier
iteration of the same duo ran `cortex` trimmed to 64K so `senses` could hold
its full native 128K instead; the current default flips that trade-off in
`cortex`'s favor (see [`docs/colleague-stack.md`](colleague-stack.md#migration-before--after)
for the full before→after migration table, including the legacy single-model
scaffold's 256K). Serving `cortex` **solo** (no multimodal gear) restores its
load-tested full-256K/util-0.6 headroom (~75 GiB). The prior co-resident dense
**24B** Mistral *generate-fallback* was removed (two ~30B NVFP4 models do not
co-fit a shared GB10 — see "Live validation findings" below, which predates
the Gemma duo and describes a different pairing: two ~30B-class dense/MoE
models, not the current 27B+12B duo); Mistral stays a selectable catalog
candidate (`lobes overview --list`) and the opt-in fallback example.

## Topology

```text
client / acp ──:8000──▶ model-gear-gateway   (python -m lobes.gateway)
                          │  route by `model` / task family / tier alias
                          ├──▶ model-gear-vllm-primary    :8000  generate main tier (→ failover if a fallback is wired)
                          ├──▶ model-gear-vllm-multimodal :8000  generate multimodal tier (vision+audio)
                          ├──▶ model-gear-vllm-embed      :8000  embed (/v1/embeddings)
                          └──▶ model-gear-vllm-rerank     :8000  score/rerank (/v1/rerank, /v1/score)
```

Five containers by default, all `restart: unless-stopped`:

| Container | Role | Host port |
|---|---|---|
| `model-gear-gateway` | stdlib reverse proxy (the single OpenAI front) | `${VLLM_PORT:-8000}` |
| `model-gear-vllm-primary` | generate `main` tier (default: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`) | internal only |
| `model-gear-vllm-multimodal` | generate `multimodal` tier (`coolthor/gemma-4-12B-it-NVFP4A16`, vision+audio, native MTP) | internal only |
| `model-gear-vllm-embed` | embedding gear (`Qwen/Qwen3-Embedding-0.6B`, `/v1/embeddings`) | internal only |
| `model-gear-vllm-rerank` | reranker gear (`Qwen/Qwen3-Reranker-0.6B`, `/v1/rerank` + `/v1/score`) | internal only |

The backends are reachable only on the compose network (`http://vllm-primary:8000`,
`vllm-embed:8000`, `vllm-rerank:8000`); only the gateway is published to the host. The
gateway needs no Docker socket access — compose owns the lifecycle; the gateway
only routes.

Each vLLM gear runs through `mg-logwrap` so its output (and any crash trace)
persists to per-boot files under the host log dir and **survives restart/recreate** —
read them with `lobes logs {primary,embed,rerank}` even after a container is gone.
See [docs/durable-logs.md](durable-logs.md) (issue #50).

### Engine: one vLLM nightly across the default fleet

Since the fleet-wide nightly-unification migration
(`docs/vllm-nightly-migration.md` §4–§8), the four default-on gears —
`vllm-primary`, `vllm-multimodal`, `vllm-embed`, and `vllm-rerank` — all pin
the **same** vLLM nightly digest
(`vllm/vllm-openai@sha256:7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9`,
vLLM `0.23.1rc1.dev672`) that the Gemma multimodal gear already ran before the
migration. `vllm-primary`/`vllm-embed`/`vllm-rerank` pull that digest
directly; `vllm-multimodal` (and the opt-in `vllm-multimodal-coder`) build it
via `Dockerfile.vllm-gemma4` (needed for the native `gemma4_unified` class +
audio extras) — same base image, different Dockerfile. One engine, fleet-wide,
for every gear a caller reaches through the gateway by default; a same-engine
27B-vs-12B comparison (`docs/vllm-nightly-migration.md` §6) is no longer
confounded by engine version.

**Opt-in gears still pin the pre-migration NGC image.** `vllm-minor` (4B) and
`vllm-middle` (legacy 14B) — both gated behind `COMPOSE_PROFILES`, neither
default-on — still pin `nvcr.io/nvidia/vllm:26.04-py3` (vLLM `0.19.0+nv26.04`).
Migrating them is devague plan task **t8**, explicitly **parked** as a
trailing follow-up (depends on t5; per its acceptance criteria, "if a gear
cannot serve on nightly its residual is documented, not silently dropped").
Until t8 lands, activating `minor` or `middle` runs two engines side by side
(nightly for the default gears, NGC 26.04 for the opt-in ones) — each gear is
independent so this is functionally fine, just worth knowing when diagnosing
version-specific behavior. This split lives in the **templates**
(`lobes/templates/fleet/`); a live redeploy (`lobes fleet up --apply`) is what
actually activates a template change on a running fleet.

### Adding a fallback

The gateway adds a second backend **only** when `FALLBACK_URL` or
`FALLBACK_SERVED_NAME` is set. To add a warm fallback: define a `vllm-fallback`
service in the fleet compose (mirror `vllm-primary` with the fallback's model /
quantization / tokenizer / tool-parser), add it to the gateway's `depends_on`,
set `FALLBACK_URL` + `FALLBACK_SERVED_NAME` on the gateway, and **drop both
`*_GPU_MEM_UTIL` values** so they sum well under 1.0. The archived dense Mistral
fallback config is in git history and
[`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md).

### Minor co-resident companion (opt-in)

The fleet compose also ships a `vllm-minor` service under `profiles: [minor]` —
a small 4B bf16 companion generate model that co-resides with the primary.
Activate it by setting `COMPOSE_PROFILES=minor` in `.env` or passing
`--profile minor` to `docker compose`. The gateway routes requests for
`model: Qwen/Qwen3.5-4B` to this backend only when `MINOR_BASE_URL` and
`MINOR_SERVED_NAME` are set in the gateway's environment; the defaults are empty,
so the gateway ignores the minor backend unless the operator explicitly opts in.

At `VLLM_MINOR_GPU_MEM_UTIL=0.10` (~13 GiB) the 4B model co-resides alongside
the 27B primary (~75 GiB) within the 128 GB GB10 budget, with the two ~0.6B
gears (util 0.06 each) also co-resident. See
[`docs/qwen3.5-4b-minor.md`](qwen3.5-4b-minor.md) for governance, the
minor-lobe verbs (`lobes run minor`, `lobes route`, `lobes eval minor`), and
serving details.

### Generate-lane tier aliases

The gateway supports three capability-tier **aliases** for the generate lane.
Callers send `model=main|minor|multimodal` (or the back-compat aliases
`hard|cheap|normal`) instead of a full model id; the gateway resolves to the
appropriate warm backend:

| Alias | Back-compat alias | Role | Checkpoint | Notes |
|---|---|---|---|---|
| `main` | `hard` | `primary` | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | full text capability; default-on |
| `minor` | `cheap` | `minor` | `Qwen/Qwen3.5-4B` | fast, small-brain; opt-in (`--profile minor`) |
| `multimodal` | `normal` | `multimodal` | `coolthor/gemma-4-12B-it-NVFP4A16` | text+image+audio, native MTP; default-on |
| `multimodal-coder` | — | `candidate` (opt-in) | `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` | coding-strong; opt-in `--profile multimodal-coder`, reachable via its own alias once wired (not a tier) |

**Colleague-role aliases (issue #81):** `model=cortex` and `model=senses` are
additional aliases for `main`/`hard` and `multimodal`/`normal` respectively —
same backends, same fallback contract, just the Colleague-facing role name
(`cortex` = the reasoning/decision authority, `senses` = perception/intake).
`minor` has no role-name alias — it is not one of the six first-class
Colleague roles; it is the servable floor under pressure (an explicit `minor`
request is always served, while full tiers are shed — see "Pressure policy and
busy backpressure" below). See [`docs/colleague-stack.md`](colleague-stack.md)
for the full six-role contract (`cortex`/`senses`/`embedder`/`reranker`/`stt`/`tts`),
their `responsibilities`/`forbidden_responsibilities`, and `GET /capabilities`.

**Fallback contract:** when a tier's own backend is absent, the alias falls back
**upward** to the nearest available generate tier. `minor`→primary when the minor
gear is not started; `main` always resolves to the primary (which is always warm).
`multimodal` falls back to `main` if the multimodal gear is not wired. Pooling gears
(`embed` / `rerank`) are never reached via tier aliases — they are
task-family-routed, not tier-routed.

The `minor` (4B) backend and the legacy `middle` (14B) backend are opt-in compose
profiles (`--profile minor` / `--profile middle`). Uncomment their `*_BASE_URL` in
`.env` after activating. Both still run the pre-migration NGC image (26.04-py3 /
vLLM 0.19.0) — see "Engine" above; migrating them to nightly is the parked task
t8. See [`docs/qwen3-14b-nvfp4.md`](qwen3-14b-nvfp4.md) for the legacy 14B
serving details and [`docs/gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md) for the
multimodal gear.

**Caller migration example:** switch from a hardcoded model id to a tier alias
with no other change:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="ignored")

# Before — hardcoded model id (breaks if the primary is swapped)
response = client.chat.completions.create(
    model="sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
    messages=[{"role": "user", "content": "Summarise this PR in one sentence."}],
)

# After — tier alias (gateway resolves to the right gear; survives primary swap)
response = client.chat.completions.create(
    model="main",   # or "multimodal" / "minor"
    messages=[{"role": "user", "content": "Summarise this PR in one sentence."}],
)
```

The same swap works with raw `curl`:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "multimodal", "messages": [{"role": "user", "content": "Hello"}]}'
```

**LoRA scope:** LoRA adapter training targets the 4B bf16 `minor` lobe
(`Qwen/Qwen3.5-4B`) — not the 14B or 27B. The 14B NVFP4 and the Gemma 4 12B
multimodal are inference-only; there is no `lobes train` verb.

### Pressure policy and busy backpressure

The gateway enforces a **memory-pressure policy** that **sheds** full-tier
requests when the host is under swap or I/O pressure — instead of silently
degrading them onto a different model. The policy is side-effect-free and purely
computed from `/proc` readings (`lobes.gateway._pressure_policy`).

Under pressure a `main`/`cortex` or `multimodal`/`senses` request is shed with
**HTTP 429 + `Retry-After`** ("busy, retry shortly") — the gateway never
substitutes a cheaper or different-capability model in its place (issue #85). An
explicit `minor` request is the floor and is **always served** (served as
requested, not a substitution). This replaced the former degrade-to-minor
behaviour: there is no `LOBES_PRESSURE_POLICY` toggle and no silent downgrade —
which makes the old cross-capability substitution (a `cortex` request answered by
Gemma when `minor` was unwired) structurally impossible.

**Threshold table** (all comparisons are strictly `>` — exactly equal does not trigger):

| Condition | `main` / `multimodal` request | `minor` request | Mode |
|---|---|---|---|
| swap > 75 % OR iowait > 50 % | **shed → 429 busy** | served | **busy** |
| *(none)* | served as requested | served | warm |

The trigger reuses the existing swap/iowait signal — this is *not* queue-depth
admission control (considered and deferred). The old intermediate swap bands
(`LOBES_SWAP_NO_HARD_THRESHOLD`, `LOBES_SWAP_PREFER_CHEAP_THRESHOLD`,
`LOBES_IOWAIT_NO_HARD_THRESHOLD`) are retained as named env constants for
observability/tuning but no longer impose a tier ceiling.

Each threshold is a named env constant with a `LOBES_*` override:

| Env var | Default |
|---|---|
| `LOBES_SWAP_NO_HARD_THRESHOLD` | 50.0 |
| `LOBES_SWAP_PREFER_CHEAP_THRESHOLD` | 65.0 |
| `LOBES_SWAP_DEGRADED_THRESHOLD` | 75.0 |
| `LOBES_IOWAIT_NO_HARD_THRESHOLD` | 25.0 |
| `LOBES_IOWAIT_DEGRADED_THRESHOLD` | 50.0 |

**The 429 busy response.** A shed request receives:

| Field | Value |
|---|---|
| Status | `429 Too Many Requests` |
| `Retry-After` | seconds to wait before retrying (`5` by default) |
| `X-Lobes-Tier-Reason` | `busy` |
| Body | OpenAI-shaped `{"error": {"type": "server_busy", "code": "busy", "message": "…"}}` |

It is distinct from the hard **`502`** (`type: upstream_unavailable` — all
backends down, do *not* retry): a `429` means "the model is up but the box is
pressured; retry shortly." **Callers must honour `429` + `Retry-After` and retry
with backoff** — the acp `vllm-local` provider, colleague, and generic OpenAI
SDKs all treat `429` as a retryable transient. A caller that would rather wait
and retry than silently act on a weaker/wrong-capability answer is exactly who
this protects; a caller that treats `429` as fatal is strictly *less* available
under pressure than the old always-answer behaviour, so client-side retry is a
requirement, not an assumption.

**Served-path headers** (unchanged; travel with every served response,
streaming-safe — headers precede the body):

| Header | Value |
|---|---|
| `X-Lobes-Tier` | The tier actually served (`main` / `minor` / `multimodal`) |
| `X-Lobes-Tier-Reason` | `default` \| `manual_override` |

**Override header:** send `X-Lobes-Override: true` on the request to force the
gateway to serve the requested tier regardless of current pressure (the manual
escape hatch — the request is served, not shed).

**Boundary.** Only the *response* to pressure changed. The `/proc` sampler
(`lobes.runtime._pressure`) and the threshold env vars are untouched; this is not
a queue/scheduler or vLLM-batching change.

**Observability:** `lobes status --pressure` reads `/proc` live and reports the
busy-policy decision a full-tier request would receive right now — read-only, no
deployment dir or Docker needed. The gateway's `GET /status` carries the same
state in a `pressure` block:

```bash
lobes status --pressure
# mode:    busy
# shed:    main/cortex + senses requests return 429 busy (retry after 5s)
# servable: minor
# model:   Qwen/Qwen3.5-4B
# reason:  pressure
# swap:    82.4%
# iowait:  0.8%

lobes status --pressure --json
# {"mode": "busy", "shed": true, "servable_tier": "minor", "model": "...",
#  "reason": "pressure", "retry_after": 5,
#  "pressure": {"swap_used_percent": 82.4, "iowait_percent": 0.8}}
```

## The gateway

A pure-stdlib (`http.server` + `http.client`, no third-party deps) reverse proxy:

- **Name routing** — a request's `model` routes to the backend that serves it,
  plus any `GATEWAY_ALIASES`. The forwarded body's `model` is rewritten to the
  backend's `--served-model-name` so the backend accepts aliased/default routes.
- **Default model** — a missing or unknown `model` routes to
  `GATEWAY_DEFAULT_MODEL` (the primary).
- **Failover** — when a fallback is wired up, a chosen backend that refuses the
  connection or returns a 5xx **before any response body** is retried against the
  other backend. (`main` and `multimodal` are different-capability tiers, not
  failover peers for each other; the embed/rerank gears are separate task families,
  not failover targets.) A 4xx is a client error (returned verbatim, no failover).
  Once a 2xx body starts streaming there is no retry — the client already has bytes.
- **Streaming** — `"stream": true` (SSE) is relayed chunk-by-chunk with per-chunk
  flushing; normal JSON is buffered with `Content-Length`.
- **Endpoints** — `/v1/chat/completions`, `/v1/completions` (generate primary),
  `/v1/embeddings` (the embedding gear), `/v1/rerank` + `/v1/score` (the reranker
  gear), `/v1/audio/transcriptions` + `/v1/audio/speech` (the `--audio` overlay
  only — fanned to the realtime bridge → Parakeet STT / Chatterbox TTS),
  `/v1/models` (OpenAI-standard, lists the loaded backend(s)),
  `/v1/models/supported` (the full supported-model catalog — every gear you can
  change to, each flagged `loaded` / `default`), `/health` (gateway liveness), and
  `/status` (the live fleet aggregate — see below).
  See [Supported catalog vs. warm backends](#supported-catalog-vs-warm-backends)
  for what `/v1/models` and `/v1/models/supported` each mean.
- **`GET /status`** — a lobes-native (non-OpenAI) JSON aggregate the gateway
  fans out to each backend's `/health` + `/metrics` and returns as
  `{object: "lobes.fleet_status", default_model, busy: {running, waiting},
  backends: [{name, task, served_name, health, metrics}], endpoints}`. The backends
  are internal-only, so the gateway is the only thing that can see them — this is
  the source for `lobes overview --live`.

The gateway image is built from the scaffolded `Dockerfile.gateway`
(`pip install lobes-cli==${MODEL_GEAR_VERSION}`, as a non-root user); `lobes init
--fleet` pins `MODEL_GEAR_VERSION` to the running lobes-cli release. The version
is required (pinning keeps the image reproducible); from-source/dev boxes that run
ahead of a PyPI release point `MODEL_GEAR_VERSION` at a published TestPyPI `.devN`
build.

### Auth (known limitation)

The gateway is a **pass-through** and is **not auth-aware** — it does not inspect
or validate `Authorization` headers. `CULTURE_VLLM_API_KEY` is enforced by vLLM on
the **single-model** serve path (`lobes serve`), but it does **not** protect the
fleet gateway's proxied endpoints (generate, embed, rerank, or `/v1/audio/*`). Keep
the gateway port off the public internet; when exposing it with `lobes tunnel`,
layer Cloudflare Access or an IP allowlist on top. Per-endpoint gateway auth is
planned for a later release.

### Supported catalog vs. warm backends

Two questions that look alike but aren't:

- **What's loaded right now?** — the model(s) actually in GPU memory. The live
  source is `GET /v1/models` (OpenAI-standard; one model in single-model mode; the
  generate primary plus the embedding + reranker gears in the fleet); `lobes fleet
  status` queries it. It changes when you
  `lobes switch` or bring the fleet up/down. (`lobes status` / `lobes whoami`
  instead report the model the deployment is *configured* to serve — from `.env` —
  plus container health, which is configuration, not a live `/v1/models` query.)
- **What's *supported* (what can I warm up)?** — the curated catalog of "gears"
  lobes knows how to serve, from `lobes overview --list` or
  `GET /v1/models/supported`. Each entry is tagged `load-tested` (proven on this
  box) or `configured` (declared, not yet proven). It's **static** — defined in
  `lobes/catalog.py`, shipped in the wheel, unchanged by what's running. On
  the gateway endpoint each entry also carries a runtime-computed `loaded` /
  `default` flag.

Mnemonic: the catalog is *what's on the menu (and which dishes we've cooked)*;
`/v1/models` is *what's hot now*.

## Verbs

```bash
lobes init --fleet --apply        # scaffold compose + .env + Dockerfile.gateway
lobes fleet up --apply            # docker compose up -d --build, wait for gateway /health
lobes fleet status                # each container's state + gateway /health + /v1/models
lobes overview --live             # live dashboard: online / offered / busy + usage + endpoints
lobes fleet down --apply          # docker compose down
```

`lobes fleet up` / `down` are **dry-run by default**; pass `--apply` to commit.
`--compose-dir` overrides the deployment dir (default `$LOBES_DIR` or
`$HOME/.lobes`). `lobes fleet status` is read-only — it reports the *warm*
backend(s) (`/v1/models`); for the full set you can switch to, use
`lobes overview --list` / `/v1/models/supported` (see above).

`lobes overview --live` is the read-only **live dashboard**: it reads the gateway
`/status` (or, against a bare single-model server, that server's `/metrics` +
`/health`) and prints what is **online** (per-backend health), **offered**
(models, task families, endpoints), **busy** (in-flight / queued requests), and
cumulative **usage** (prompt/generation tokens, finished requests by reason). HTTP-only, so it
works against a local deployment or a `lobes tunnel` hostname alike; it degrades
gracefully when a backend or its metrics is unreachable.

**`lobes switch` does not drive the fleet** — it rewrites the single-model
`VLLM_*` keys. Change the fleet primary by editing the fleet `.env`
(`PRIMARY_MODEL` and its `PRIMARY_SERVED_NAME` / `PRIMARY_GPU_MEM_UTIL`
/ `PRIMARY_TOOL_CALL_PARSER` / `PRIMARY_QUANTIZATION`) and re-running `lobes fleet
up --apply`. (A fallback, when wired up, uses the parallel `FALLBACK_*` keys.)

## Memory

The fleet default is **four always-warm backends** — the 27B generate primary
(`main` tier), the Gemma 4 12B multimodal gear (`multimodal` tier), the 0.6B
embedding gear, and the 0.6B reranker gear. The 4B `minor` (back-compat `cheap`)
and the legacy 14B are opt-in compose profiles. Default budget — the
**"always-on duo"**, live-validated co-resident on the DGX Spark GB10
2026-07-02 (`docs/vllm-nightly-migration.md` §8):

| Gear | Model | Context | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|---|---|
| `primary` (main) | `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` | **128K** (full native) | **0.30** | ~38 |
| `multimodal` (default-on) | `coolthor/gemma-4-12B-it-NVFP4A16`, native MTP on | **32K** (trimmed from 128K) | **0.14** | ~18 |
| `embed` | `Qwen/Qwen3-Embedding-0.6B` | 8K | 0.06 | ~7 |
| `rerank` | `Qwen/Qwen3-Reranker-0.6B` | 8K | 0.06 | ~7 |
| **Total (default)** | | | **0.56** | ~70 / 128 GB |

Opt-in gears (add to `COMPOSE_PROFILES`) — `minor`/`middle` still run the
pre-migration NGC image (see "Engine" above; t8 parked):

| Gear | Model | `--gpu-memory-utilization` | Approx GiB |
|---|---|---|---|
| `minor` (cheap, opt-in) | `Qwen/Qwen3.5-4B` | 0.10 | ~13 |
| `middle` (legacy, opt-in) | `nvidia/Qwen3-14B-NVFP4` | 0.12 | ~15 |
| `multimodal-coder` (opt-in) | `sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4` | 0.12 | ~15 |

The **primary now serves its full 128K native context**
(`PRIMARY_MAX_MODEL_LEN=131072`, `PRIMARY_GPU_MEM_UTIL=0.30` — util-bound, not
context-bound, so the earlier 64K trim was not needed to hold this util) while
the default-on multimodal gear is **trimmed to 32K context**
(`MULTIMODAL_MAX_MODEL_LEN=32768`, `MULTIMODAL_GPU_MEM_UTIL=0.14`, down from an
earlier 128K/0.22 pairing) to free the KV headroom the primary's full context
needs (see "Always-on duo budget" below). Without the
multimodal gear (single-primary mode), restore `PRIMARY_GPU_MEM_UTIL=0.6` and
optionally `PRIMARY_MAX_MODEL_LEN=262144` for the full 256K solo footprint
(the load-tested default; see findings below).

### Always-on duo budget (live-validated, 2026-07-03)

Can `cortex` serve **128K** *and* `senses` serve **32K** co-resident on one GB10,
without either starving the other? **Yes** — live-validated on the DGX Spark GB10
2026-07-03 (#81 t12): `cortex` (27B MTP @128K, util 0.30) held **3.58×** measured
concurrency (18.02 GiB / 468,886-token KV cache) and `senses` (Gemma 4 12B @32K,
util 0.14) held **5.62×** (8.87 GiB / 184,084-token KV) — both healthy and
simultaneous, alongside embed + rerank (util 0.06 each). Default budget
`0.30 + 0.14 + 0.06 + 0.06 = 0.56`.

This confirms the 27B KV is **util-bound, not context-bound**: the prior
2026-07-02 pairing served `cortex` at 64K/util 0.30 (**6.36×**) and `senses` at
128K/util 0.22 (**4.67×**); the #81 rebalance to 128K/32K at the *same* primary
util simply trades cortex concurrency (6.36×→3.58×) for the longer context while
freeing KV headroom on the senses side (32K at util 0.14 keeps 5.62×). It
supersedes the #71 co-resident-safe fallback (8K context @ util 0.12 for the
multimodal gear — see
[`gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md#live-validation-status-71)) that
predated the duo-budget retune.

The co-resident embedding and reranker gears are ~0.6B each at `*_GPU_MEM_UTIL=0.06`
(a couple GiB apiece), so they tuck into the remaining headroom without crowding
the primary; what does **not** co-fit is a second ~30B *generate* model (below).
That still leaves room for the OS and other processes.

`--gpu-memory-utilization` is a fraction of *total* unified memory, computed
independently per vLLM process (they don't coordinate). So **if you add a warm
fallback**, `PRIMARY_GPU_MEM_UTIL` + `FALLBACK_GPU_MEM_UTIL` must sum well under
1.0 — two ~30B NVFP4 models do **not** co-fit a GB10 that is also running other
services (the prior `0.40` + `0.35` co-residence default OOM-looped; that's why
the fallback was removed). **Validate live** (watch `spark memory` / `nvidia-smi`
at `lobes fleet up`; OOM is the top operational risk).

Note the throughput trade-off if you do co-resident two backends: decode is
memory-bandwidth bound and the bandwidth (~273 GB/s) is **shared** — two backends
decoding *simultaneously* split it. The gateway routes one request to one backend,
so a single client sees full speed.

## Live validation findings — DGX Spark (GB10), 2026-05-30

First live `lobes fleet up` of the 27B-primary + 35B-A3B-fallback pair on
`spark-f8a9` (a **shared** box: tritonserver/realtime-api, nova, reachy, mongo
also running, ~12–20 GiB baseline). Measured with `dgx-spark-cli` (`spark`):

| What | Result |
|---|---|
| **27B (primary) solo load → `/health`** | **~423 s (~7 min)**: weight load 160 s (28.25 GiB), profiling/warmup 55 s, CUDA-graph capture + KV ~200 s |
| 27B decode (batch=1, 512 tok) | **8.0 tok/s**; prefill 2,015 tok in 3.29 s |
| 27B footprint | **~75.5 GiB at util 0.6** (≈ 28 GiB weights + 42 GiB KV + 3.7 GiB CUDA graphs) |
| **35B-A3B (old fallback) load** | **Did not complete.** Co-resident: `CUDA error: out of memory` on engine init → 14+ restart crash-loop. Even *solo* (65 GiB free): crashed/stalled at "Loading safetensors 0%", never `/health` in 8+ min. No benchmark obtained. |
| Co-residence (27B + 35B-A3B) | **Not viable on this box.** 27B alone (~75 GiB) + 35B-A3B (~24 GiB weights + KV) + baseline services exceed the 121.7 GiB unified pool → OOM + swap thrash (swap hit 68 %). |
| **Mistral-24B (new fallback) solo load → `/health`** | **Loaded cleanly** (port 8001, util 0.4): 15.05 GiB weights, 30.69 GiB KV, ~49.6 GiB total. Decode **14.9 tok/s**; prefill 2,009 tok in 1.49 s; tool calling ✅. See [`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md). |

**Conclusion (2026-05-30, scoped to this pairing) — the "two always-warm ~30B
*generate* models" premise needs a dedicated box.** On a GB10 shared with other
services, two ~30B NVFP4 models do not co-fit with usable KV caches. At the
time, the default fleet therefore served the **Qwen generate primary** at its
load-tested solo headroom (util 0.6, full 256K, ~75 GiB), with the tiny
embedding + reranker gears co-resident (util 0.06 each). If you genuinely need
two *~30B-class* warm generate models, run on a dedicated machine, pair two
small models, or wire the opt-in fallback (see "Adding a fallback") and drop
both utils. Single-model `lobes switch` (one warm at a time) remains the other
path for that case.

**Superseded by the always-on duo (2026-07-02).** This conclusion is about
pairing the 27B primary with a *second ~30B-class* model (35B-A3B / 24B
Mistral) — it does not rule out the Gemma 4 12B multimodal gear, which is
smaller and retuned for co-residency (see "Always-on duo budget" above). The
default fleet today runs **two** generate backends — `cortex` (the 27B
primary; **128K**, util 0.30) and `senses` (the 12B Gemma multimodal gear;
**32K**, util 0.14) — live-validated co-resident on this same GB10, per
`docs/vllm-nightly-migration.md` §8. (An earlier retune of this same duo ran
`cortex` at 64K/util 0.30 and `senses` at its full 128K/util 0.22 before the
current context rebalance flipped the trade-off in `cortex`'s favor — see
[`docs/colleague-stack.md`](colleague-stack.md#migration-before--after) for
the full before→after table.) The "one generate backend" constraint above
still applies to a *second ~30B-class* warm fallback (the opt-in path), not to
the `senses` gear.

**Fallback history.** The original 35B-A3B MoE fallback never loaded
([`docs/qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md)); it was replaced
(2026-05-30) by the dense `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
(loads reliably, ~15 GiB weights —
[`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md)). Even the
dense 24B stayed tight on a shared box, so the warm fallback was **removed from
the default fleet** — Mistral remains a selectable catalog candidate and the
documented opt-in fallback. The `0.55`/`0.30` → `0.40`/`0.35` util history above is
the record of that co-residence struggle.

## Coherence with the single-model verbs

The fleet `.env` mirrors `VLLM_MODEL` / `VLLM_SERVED_NAME` / `VLLM_TOOL_CALL_PARSER`
(= the primary's) so the read-only single-model verbs (`lobes status`,
`lobes whoami`, `lobes doctor`'s `env_coherence` check) stay sensible on a fleet
deployment. `culture.yaml`'s `model: vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`
resolves through the gateway on `:8000` as the default.
