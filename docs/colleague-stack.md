# The Colleague stack: six roles, one contract

> The six first-class, Colleague-facing roles lobes exposes over the fleet —
> `cortex` / `senses` / `embedder` / `reranker` / `stt` / `tts` — how a caller
> discovers them, drives them, measures them, and the before→after context
> migration that shipped alongside this contract (issue #81).

This doc is the **role contract** reference. For the fleet's Docker topology,
tuning knobs, and memory budget, see [`docs/gateway-fleet.md`](gateway-fleet.md);
for the raw OpenAI wire endpoints (`/v1/chat/completions`, `/v1/embeddings`, …)
see [`docs/openai-api.md`](openai-api.md).

## Why roles, not model ids

Before issue #81, a Colleague client that wanted the vision-capable gear had to
know its literal served name (`coolthor/gemma-4-12B-it-NVFP4A16`) and hardcode
it. That breaks the moment an operator swaps the checkpoint. The **role**
vocabulary fixes this: a client asks for a *capability* (`cortex` — reasoning
and decisions; `senses` — perception and intake) and lobes resolves it to
whichever concrete model, endpoint, and context that capability currently
serves. Renaming or re-quantizing the underlying checkpoint is then an
operator-side change with **zero client-code change** — see "Client flow"
below.

## The six roles

| Role | Backend / service | Endpoint path | What it's for |
|---|---|---|---|
| `cortex` | `primary` (generate) | `POST /v1/chat/completions` | Reasoning, deciding, planning, tool use, repo actions — the final authority. |
| `senses` | `multimodal` (generate) | `POST /v1/chat/completions` | Intake/perception (text+image) and speaking back to the user. Does **not** decide or act. |
| `embedder` | `embed` (pooling) | `POST /v1/embeddings` | Dense text embeddings for memory/retrieval. |
| `reranker` | `rerank` (pooling) | `POST /v1/rerank` (+ `/v1/score`) | Reordering/scoring retrieved candidates. |
| `stt` | Parakeet (audio overlay, opt-in) | `POST /v1/audio/transcriptions` | Speech-to-text. |
| `tts` | Chatterbox (audio overlay, opt-in) | `POST /v1/audio/speech` | Text-to-speech. |

> **`senses` is vision-only intake — audio is not currently served (issue
> #101).** The `coolthor/gemma-4-12B-it-NVFP4A16` checkpoint behind `senses`
> declares an `audio_config` in its own model config, but on this vLLM serving
> path (`gemma4_unified`) an `input_audio` content part is silently **dropped**
> rather than rejected: a caller gets `200 OK` and a fluent answer that ignored
> the audio. Live evidence and the tracking issue are in
> [`docs/gemma-4-12b-nvfp4.md`](gemma-4-12b-nvfp4.md#live-validation-status-71).
> For speech, use the purpose-built **`stt`** role (Parakeet, `POST
> /v1/audio/transcriptions`) instead — it remains first-class and is
> unaffected by this gap.

`cortex`, `senses`, `embedder`, and `reranker` are always enumerated (present
with `loaded=false` if their gear isn't wired in this deployment); `stt`/`tts`
require `lobes init --fleet --audio`. **`brain` is not a valid role name** —
`cortex` is the only reasoning/decision role.

### Responsibilities and forbidden responsibilities

Each role carries a declared division of labour — what it is expected to own,
and (for `senses`) what it must **not** do. These are **runtime-descriptor
tokens, not correctness claims** — lobes does not grade whether a role did its
job well; that judgment is Colleague's (see "Runtime-only, always" below).

| Role | `responsibilities` | `forbidden_responsibilities` |
|---|---|---|
| `cortex` | `reasoning`, `deciding`, `planning`, `tool_use`, `code_repo_actions`, `validation`, `final_authority` | *(none — cortex is the final authority)* |
| `senses` | `intake`, `normalize_input`, `classify_intent`, `prepare_context_packet`, `speak_back` | `final_decision`, `repo_action`, `security_decision` |
| `embedder` | `vectorization`, `memory_retrieval_input` | *(none)* |
| `reranker` | `retrieval_ordering`, `relevance_refinement` | *(none)* |
| `stt` | `transcribe`, `audio_input_to_text` | *(none)* |
| `tts` | `speech_output`, `synthesize` | *(none)* |

## cortex/senses ↔ primary/multimodal — one mapping, three vocabularies

`cortex` and `senses` are **new names layered on the existing `primary` /
`multimodal` backends and the `main` / `multimodal` capability tiers** — no
internal service, container, or env var was renamed. All three vocabularies
resolve to the same backend (`lobes/catalog.py`'s `TIER_ROLE`):

| Backend (`role_hint`) | Primary tier alias | Back-compat tier alias | Colleague role name |
|---|---|---|---|
| `primary` | `main` | `hard` | `cortex` |
| `multimodal` | `multimodal` | `normal` | `senses` |
| `minor` | `minor` | `cheap` | *(no role name — `minor` has no Colleague role; it's the servable floor under pressure, not a first-class capability)* |

A caller can send `model=cortex`, `model=main`, or `model=hard` to
`/v1/chat/completions` and reach the exact same warm backend. **All the old
aliases keep working** — this is additive vocabulary, not a rename. See
[`docs/gateway-fleet.md`](gateway-fleet.md#generate-lane-tier-aliases) for the
full tier-alias fallback contract (busy backpressure under pressure,
`multimodal` falling back to `main` when unwired, etc.) — that mechanism is
unchanged by the role layer.

## Discovery: `GET /capabilities` and `lobes capabilities` / `lobes endpoint`

A client that wants to drive any role needs exactly **one** thing: the fleet's
base URL. Everything else — which model backs a role, whether it's loaded,
what context it's served at — comes from the contract itself.

```bash
lobes capabilities              # human-readable table, all six roles
lobes capabilities --json       # the machine-readable contract
lobes endpoint cortex           # just the base URL for one role
curl -s http://localhost:8000/capabilities   # the same contract, over HTTP
```

`lobes capabilities` and `GET /capabilities` are **the same payload** — both
are built by the one canonical registry builder,
`lobes.roles.build_role_registry` (`lobes/roles.py`), so there is exactly one
source of truth for the role→endpoint contract. The CLI reads the deployment's
`.env` off disk (soft-resolved — an unscaffolded deployment still answers,
with every role but `cortex` reported `loaded=false`); the gateway reads its
own container environment.

### JSON contract shape

`GET /capabilities` (and `lobes capabilities --json`) returns an object keyed
by role name, each value carrying exactly these fields:

```text
{
  "<role>": {
    "role": str,                          # "cortex" | "senses" | "embedder" | "reranker" | "stt" | "tts"
    "model": str,                         # the served model id this role resolves to (never blank)
    "runtime": str,                       # "vllm" | "parakeet" | "chatterbox"
    "endpoint": str,                      # client-reachable base URL to dial ("" when not wired)
    "path": str,                          # the OpenAI path, e.g. "/v1/chat/completions"
    "context": int,                       # SERVED context in tokens (deployment override, else catalog native)
    "quant": str,                         # vLLM quantization; "" for pooling/audio roles
    "mtp": bool,                          # speculative decoding (MTP draft head) active
    "responsibilities": [str, ...],
    "forbidden_responsibilities": [str, ...],
    "ready": bool | null,                 # see the note below
    "loaded": bool                        # is this role's backend wired in THIS deployment?
  },
  ...
}
```

**Every role's `endpoint` is the one client-reachable gateway origin** — dial
it directly (issue #87). All six roles (`cortex`/`senses`/`embedder`/`reranker`
**and** `stt`/`tts`) report the same base URL because routing happens via the
`model` field / the OpenAI `path`, not distinct per-role URLs; the internal
upstream hosts (`vllm-primary:8000`, `realtime:8080`) are never leaked. When you
fetch `GET /capabilities`, the gateway advertises the origin **you actually
dialed** (from the request `Host` header), so `endpoint` is reachable as-is; set
`GATEWAY_PUBLIC_URL` to override it for a tunnel / Host-rewriting reverse proxy.
A role is `""` only when it isn't wired (e.g. `stt`/`tts` without `--audio`).

**`ready` differs by transport, deliberately.** `lobes capabilities --json`
reports a **configured** signal (`ready == loaded`) — a read-only CLI on the
host can't reach the internal backends to probe them, so it doesn't try; use
`lobes measure` for a CLI-side live probe. `GET /capabilities` is the honest
one for consumers: for `stt`/`tts` it now reports a **live** readiness probe of
the audio backend (issue #89) — `ready: true` only when an audio round-trip
would actually succeed (Chatterbox + Parakeet both up, no poisoned CUDA
context), `false` while they warm — so an advertised-ready audio role is truly
consumable. The four gateway-fronted roles still report `ready` as a
same-cost-as-`loaded` boolean. No `ready` value is a task-quality claim.

Example (`cortex`, fully wired, default fleet):

```json
{
  "role": "cortex",
  "model": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
  "runtime": "vllm",
  "endpoint": "http://localhost:8000",
  "path": "/v1/chat/completions",
  "context": 131072,
  "quant": "modelopt",
  "mtp": true,
  "responsibilities": ["reasoning", "deciding", "planning", "tool_use", "code_repo_actions", "validation", "final_authority"],
  "forbidden_responsibilities": [],
  "ready": true,
  "loaded": true
}
```

An unwired role (e.g. `stt`/`tts` without `--audio`, or `senses` before the
multimodal gear is up) is **never omitted** — it's returned with
`loaded: false` and the model it *would* serve named from the catalog, so a
client can always render all six roles.

## Serving: `lobes up <role>` and `colleague-stack`

`lobes up` starts (or, with `--down`, stops) **one** role's gear without
touching the rest of the fleet:

```bash
lobes up cortex --apply             # docker compose up -d vllm-primary
lobes up senses --apply             # docker compose up -d vllm-multimodal
lobes up embedder --apply           # docker compose up -d vllm-embed
lobes up reranker --apply           # docker compose up -d vllm-rerank
lobes up stt --apply                # requires the --audio overlay
lobes up tts --apply                # requires the --audio overlay
lobes up colleague-stack --apply    # all six roles at once (requires --audio scaffolded)
```

Dry-run by default (prints the exact `docker compose …` command); `--apply`
commits. `colleague-stack` is a first-class bundle — the four default fleet
roles **plus** the audio overlay's `stt`/`tts` — not a compose `profiles:`
tag, because tagging the already-default-on services with a profile would
demote them out of the default fleet (a regression). If the audio overlay
isn't scaffolded, `colleague-stack` (and `up stt`/`up tts`) fail with a
remediation pointing at `lobes init --fleet --audio --apply`, rather than
silently starting only four of the six roles.

## Measurement: `lobes measure`

Read-only, per-role **runtime** metrics — never a task-quality or correctness
claim (lobes measures serving performance; whether an *answer* was good is
Colleague's call):

```bash
lobes measure              # all six roles, table
lobes measure --json       # all six roles, JSON
lobes measure --role cortex --json
```

Metrics are grouped by the role's family:

- **LLM roles** (`cortex`, `senses`): `ttft_ms`, `decode_tps`, `prefill_tps`,
  `context`, `mem_usage_pct` (when the vLLM `/metrics` scrape is cheaply
  reachable); `restart_count`/`error_count` are always `null` (not cheaply
  available without a docker inspect, which this verb deliberately never does).
- **Pooling roles** (`embedder`, `reranker`): `reqs_per_sec`, `docs_per_sec`,
  `latency_ms`, `batch_size`, `loaded`.
- **Audio roles** (`stt`, `tts`): `rtf` (real-time factor), `latency_ms`,
  `duration_ms`, `failure_rate`.

An unloaded or unreachable role degrades to `ready: false` with every metric
`null` — this is the normal case in CI or on a partial deployment, not an
error; a dead `senses` backend never stops `cortex` from being measured.

## Cross-role comparison: `lobes benchmark --profile`

`lobes benchmark --profile <name>` runs a RUNTIME-ONLY, side-by-side
comparison across a fleet *profile* — built on the same per-role probes
`lobes measure` uses, not the load-test engine `--all-lobes` uses:

| Profile | What it compares |
|---|---|
| `cortex-only` | The `cortex` generate lane alone. |
| `cortex+senses` | `cortex` and `senses` side by side (the always-on duo). |
| `senses-direct` | `senses` addressed directly (cheap/front-door tasks). |
| `qwen-nvfp4-vs-bf16` | The current `cortex` endpoint probed as both a quantized and an unquantized catalog variant — reported `available: false` with a `reason` when the catalog doesn't carry both sides (never fabricated). |
| `all` | Every profile above. |

```bash
lobes benchmark --profile cortex+senses --json
lobes benchmark --profile all
```

Degrades gracefully offline: an unreachable role or a catalog-missing variant
is reported unavailable with a `reason`, never a crash or an invented number.

## Client flow: base URL → `/capabilities` → drive by role

A Colleague client needs **only the fleet's base URL**. The whole discovery
and dispatch flow:

1. `GET <base_url>/capabilities` once.
2. Read the role you want (`cortex`, `senses`, `embedder`, `reranker`, `stt`,
   `tts`) out of the response — its `endpoint`, `model`, and `path`.
3. `POST <endpoint><path>` with `"model": <model>` and the role-appropriate
   body shape (chat messages for `cortex`/`senses`, `input` for `embedder`,
   `query`+`documents` for `reranker`).

No model id is ever hardcoded in the client. Concretely (Python, stdlib-only):

```python
import json
import urllib.request

def call_role(base_url: str, role: str, body_extra: dict) -> dict:
    with urllib.request.urlopen(base_url.rstrip("/") + "/capabilities") as r:
        contract = json.load(r)
    info = contract[role]
    body = {"model": info["model"], **body_extra}
    req = urllib.request.Request(
        info["endpoint"].rstrip("/") + info["path"],
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)

call_role("http://localhost:8000", "cortex",
          {"messages": [{"role": "user", "content": "ping"}]})
```

Because `cortex` and `senses` are **gateway-fronted**, they (along with
`embedder`/`reranker`) share the **same `endpoint`** — the gateway's base
URL — and routing between them happens purely via the `model` field the
contract handed back. `stt`/`tts` resolve to the audio-overlay bridge URL
instead.

**Rename-safety, proven.** If an operator swaps `cortex`'s served checkpoint
(`PRIMARY_SERVED_NAME` in the fleet `.env`) and re-runs `lobes fleet up
--apply`, the very next `GET /capabilities` call reflects the new model id —
the client above needs **zero code changes** to keep working. This is proven
end-to-end in `tests/test_colleague_contract.py`
(`test_colleague_follows_an_operator_rename_with_no_client_code_change`).

## Runtime-only, always

Every field lobes emits about a role — in `/capabilities`, `lobes measure`, or
`lobes benchmark --profile` — is a **serving/runtime** descriptor: latency,
throughput, context, quant, load state, or a declared (not graded)
responsibility. **Nothing lobes emits asserts answer correctness, task
quality, or agent-task success** — judging whether a *response* was actually
good is Colleague's job. This boundary is enforced by test
(`tests/test_colleague_contract.py::test_capabilities_contract_is_runtime_descriptor_only`
and `::test_measure_registry_emits_only_allowed_runtime_metric_keys`), which
scans every emitted key for quality-flavoured tokens (`accuracy`, `correct`,
`quality`, `task_success`, `success_rate`, `grade`, `score`) and fails if one
appears.

## Migration: before → after

The Colleague-role contract shipped alongside a context rebalance. Both
`cortex` and `senses` serve **less** context than the legacy single-model
scaffold did solo, in exchange for co-residency:

| Deployment shape | `cortex` / primary served context | `senses` / multimodal served context |
|---|---|---|
| **Legacy single-model scaffold** (`lobes init` / `lobes serve`, no fleet) | **256K** (`VLLM_MAX_MODEL_LEN=262144`, solo, util 0.6) | *(not served — single model only)* |
| **Fleet duo, pre-rebalance** | 64K, util 0.30 | 128K (`MULTIMODAL_MAX_MODEL_LEN=131072`, util 0.22) |
| **Fleet duo, current (this doc)** | **128K** (`PRIMARY_MAX_MODEL_LEN=131072`, util 0.30 — util-bound, not context-bound) | **32K** (`MULTIMODAL_MAX_MODEL_LEN=32768`, util 0.14) |

The pre-rebalance duo gave the vision gear its full native 128K at the
cost of trimming `cortex` to 64K; the current default flips that trade-off —
`cortex` (the final-authority reasoning role) now gets its full native 128K,
and `senses` (intake/perception) is trimmed to 32K, which is ample for the
"normalize input, classify intent, prepare a context packet" responsibilities
it's actually scoped to. Running `cortex` **solo** (no fleet, no `senses`)
still restores the legacy single-model 256K/util-0.6 footprint — the role
contract doesn't change what a solo deployment can serve, only what the
default *duo* budgets. See
[`docs/gateway-fleet.md`](gateway-fleet.md#memory) for the full budget table
and live-validation history behind this rebalance.

## See also

- [`docs/gateway-fleet.md`](gateway-fleet.md) — Docker topology, tier-alias
  fallback contract, pressure busy-backpressure policy, memory budget, live
  validation history.
- [`docs/openai-api.md`](openai-api.md) — the raw OpenAI-compatible wire
  endpoints each role sits behind.
- `lobes explain roles` — the in-CLI version of this doc.
- `lobes explain fleet` / `lobes explain gateway` — routing semantics.
- `tests/test_colleague_contract.py` — the end-to-end proof of the client flow
  and the runtime-only boundary described above.
