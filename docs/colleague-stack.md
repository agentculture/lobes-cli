# The Colleague stack: seven roles, one contract

> The seven first-class, Colleague-facing roles lobes exposes over the fleet —
> `cortex` / `senses` / `muse` / `embedder` / `reranker` / `stt` / `tts` — how
> a caller discovers them, drives them, measures them, and the before→after
> context migration that shipped alongside this contract (issue #81; `muse`
> joined as the seventh, opt-in-hosted role).

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

## The seven roles

| Role | Backend / service | Endpoint path | What it's for |
|---|---|---|---|
| `cortex` | `primary` (generate) | `POST /v1/chat/completions` | Reasoning, deciding, planning, tool use, repo actions — the final authority. |
| `senses` | `multimodal` (generate) | `POST /v1/chat/completions` | Intake/perception (text+image) and speaking back to the user. Does **not** decide or act. |
| `muse` | `muse` (generate, **opt-in hosting**) | `POST /v1/chat/completions` | Creative generation, long-form writing, ideation, a divergent second opinion. Proposes; never decides or acts. |
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

`cortex`, `senses`, `muse`, `embedder`, and `reranker` are always enumerated
(present with `loaded=false` if their gear isn't wired in this deployment —
`muse` additionally reports `feasible=false` unless a muse-hosting shape
declares it, see the note below); `stt`/`tts` require `lobes init --fleet
--audio`. **`brain` is not a valid role name** — `cortex` is the only
reasoning/decision role.

> **`muse` is an OPT-IN CORE ROLE — machine-as-brain never hosts it.** The
> `nvidia/Gemma-4-31B-IT-NVFP4` checkpoint behind `muse` is too heavy to
> co-reside with the default `cortex`+`senses` duo on a 128 GB box, so the
> default shape's hosted set stays the SIX default roles (`DEFAULT_HOSTED_ROLES`
> in `lobes/profiles/shapes.py`) while the contract set capabilities reports
> (`COLLEAGUE_ROLES`) is seven. Only an explicit muse-hosting deployment shape
> (`lobes init --shape thor-muse`) serves it — **DECLARED/UNVALIDATED** as of
> this writing: a 2026-07-17 live boot measured the budget, but the
> acceptance run/transcript is pending, #108 (see
> [`docs/gemma-4-31b-nvfp4.md`](gemma-4-31b-nvfp4.md)). On every non-hosting
> deployment `muse` is honestly `feasible: false` (and, uniquely, an unwired
> muse *defaults* to infeasible even on a stale pre-muse `.env` — see
> [`docs/gateway-fleet.md`](gateway-fleet.md#generate-lane-tier-aliases)), so
> `model=muse` 404s `role_infeasible` — referable and proxyable like every
> core role — rather than silently falling back to `cortex`.

### Responsibilities and forbidden responsibilities

Each role carries a declared division of labour — what it is expected to own,
and (for `senses` and `muse`) what it must **not** do. These are
**runtime-descriptor tokens, not correctness claims** — lobes does not grade
whether a role did its job well; that judgment is Colleague's (see
"Runtime-only, always" below).

| Role | `responsibilities` | `forbidden_responsibilities` |
|---|---|---|
| `cortex` | `reasoning`, `deciding`, `planning`, `tool_use`, `code_repo_actions`, `validation`, `final_authority` | *(none — cortex is the final authority)* |
| `senses` | `intake`, `normalize_input`, `classify_intent`, `prepare_context_packet`, `speak_back` | `final_decision`, `repo_action`, `security_decision` |
| `muse` | `creative_generation`, `long_form_writing`, `ideation`, `style_variation`, `divergent_second_opinion`, `tool_use` | `final_decision`, `repo_action`, `security_decision` — muse proposes, cortex decides |
| `embedder` | `vectorization`, `memory_retrieval_input` | *(none)* |
| `reranker` | `retrieval_ordering`, `relevance_refinement` | *(none)* |
| `stt` | `transcribe`, `audio_input_to_text` | *(none)* |
| `tts` | `speech_output`, `synthesize` | *(none)* |

Two roles carry `tool_use`: `cortex` and `muse`. They are not equivalent, and the
`forbidden_responsibilities` column is what separates them — `cortex` may act on a
tool result (`repo_action`, `final_authority`); `muse` may only *research* with
one. muse calling `read_file` to ground a proposal is in-contract; muse calling
anything that writes is not. `senses` has no `tool_use` at all: it is
intake/perception, even though its Gemma lane *can* serve tool calls (see `tools`,
below — a capability of the lane, not a licence for the role).

## cortex/senses ↔ primary/multimodal — one mapping, three vocabularies

`cortex` and `senses` are **new names layered on the existing `primary` /
`multimodal` backends and the `main` / `multimodal` capability tiers** — no
internal service, container, or env var was renamed. All three vocabularies
resolve to the same backend (`lobes/catalog.py`'s `TIER_ROLE`):

| Backend (`role_hint`) | Primary tier alias | Back-compat tier alias | Colleague role name |
|---|---|---|---|
| `primary` | `main` | `hard` | `cortex` |
| `multimodal` | `multimodal` | `normal` | `senses` |
| `muse` | `muse` | *(none — new with the role)* | `muse` — the first role whose name IS the backend/tier name; capability order is `minor` < `multimodal` < `muse` < `primary` |
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
lobes capabilities              # human-readable table, all seven roles
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
    "role": str,                          # "cortex" | "senses" | "muse" | "embedder" | "reranker" | "stt" | "tts"
    "model": str,                         # the served model id this role resolves to (never blank)
    "runtime": str,                       # "vllm" | "parakeet" | "chatterbox"
    "endpoint": str,                      # client-reachable base URL to dial ("" when not wired)
    "path": str,                          # the OpenAI path, e.g. "/v1/chat/completions"
    "context": int,                       # SERVED context in tokens (deployment override, else catalog native)
    "quant": str,                         # vLLM quantization; "" for pooling/audio roles
    "mtp": bool,                          # speculative decoding (MTP draft head) active
    "tools": bool,                        # does this endpoint accept OpenAI `tools`? see the note below
    "responsibilities": [str, ...],
    "forbidden_responsibilities": [str, ...],
    "ready": bool | null,                 # see the note below
    "loaded": bool,                       # is this role's backend wired in THIS deployment?
    "feasible": bool,                     # can THIS MACHINE serve this role at all? (deployment-shapes)
    "hosted_by": str,                     # OPTIONAL — present only when feasible=false and a peer origin is declared
    "proxied": bool                       # OPTIONAL — present (and true) only when this box also forwards to that peer
  },
  ...
}
```

`feasible` is always present (a hardware/deployment fact — see
[`docs/deployment-shapes.md`](deployment-shapes.md)); `hosted_by` and
`proxied` are **optional keys**, added only for a role this box does not
host, and only when the operator declared a peer for it — see
[A third role state: proxied](#a-third-role-state-proxied) below for the full
three-state contract.

**`tools`** answers "can I put an OpenAI `tools` array on a request to this
role?" — `true` for the three generate lobes (`cortex`/`senses`/`muse`), `false`
for `embedder`/`reranker` (pooling lanes, no chat endpoint) and `stt`/`tts`. It
is a fact about the MODEL the role resolves to, derived from the catalog's
`tool_parser` — the same field the served `--tool-call-parser` flag is built
from — so it reports `true` for a role this box does not host, exactly like
`model`/`context`/`quant`/`mtp` do; `feasible`/`ready` are what tell you whether
you can reach it. It is deliberately a bool rather than the parser's name: the
served parser can legitimately diverge from the catalog's (the primary lane
defaults to the `qwen3_coder_thinking` *plugin* over the catalog's base
`qwen3_coder`, and `PRIMARY_TOOL_CALL_PARSER` can override it), so naming one
here would be a claim `lobes.roles` cannot honestly make — while *whether* tools
are accepted does not vary under that divergence. Like every field in this
contract it is runtime-only: it says the endpoint accepts `tools`, never that a
given call will be correct or succeed.

Note `tools` and `tool_use` are different questions, and a role can have one
without the other. `tools` is a CAPABILITY of the lane; `tool_use` is a
RESPONSIBILITY of the role. `senses` has `tools: true` and no `tool_use`: its
Gemma lane can serve tool calls, but the division of labour doesn't ask it to.

**Every role's `endpoint` is the one client-reachable gateway origin** — dial
it directly (issue #87). All seven roles (`cortex`/`senses`/`muse`/`embedder`/
`reranker` **and** `stt`/`tts`) report the same base URL because routing happens via the
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
consumable. The gateway-fronted roles (`cortex`/`senses`/`muse`/`embedder`/
`reranker`) still report `ready` as a same-cost-as-`loaded` boolean **unless
the role is proxied** (below), in which case `ready` reflects a live probe of
the *peer*, not a local boolean. No `ready` value is a task-quality claim.

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
  "tools": true,
  "responsibilities": ["reasoning", "deciding", "planning", "tool_use", "code_repo_actions", "validation", "final_authority"],
  "forbidden_responsibilities": [],
  "ready": true,
  "loaded": true
}
```

An unwired role (e.g. `stt`/`tts` without `--audio`, or `senses` before the
multimodal gear is up) is **never omitted** — it's returned with
`loaded: false` and the model it *would* serve named from the catalog, so a
client can always render all seven roles. (An unwired `muse` additionally
defaults to `feasible: false` — the opt-in-hosting honesty rule above.)

## A third role state: proxied

A role's `feasible: false` (this box's deployment shape dropped it — see
[`docs/deployment-shapes.md`](deployment-shapes.md)) has always meant one of
two things a client can tell apart by key presence alone:

- **referral-only** — `hosted_by: "<peer origin>"` is present, `proxied` is
  **absent** (never `false` — a key that doesn't apply is omitted, not set to
  a falsy sentinel). The caller must dial the peer origin itself; this box
  answers the role with `404 role_infeasible`.
- **proxied** — `hosted_by` **and** `proxied: true` are both present. This box
  has opted in to *following its own referral* (issues #115/#127): a request
  for the role is forwarded to the peer named in `hosted_by`, and the answer
  comes back through this box's own `endpoint` — the caller never has to
  learn the peer exists or change its request.

```json
{
  "role": "senses",
  "model": "coolthor/gemma-4-12B-it-NVFP4A16",
  "endpoint": "http://localhost:8000",
  "feasible": false,
  "hosted_by": "http://thor.example.ts.net:8000",
  "proxied": true,
  "ready": true,
  "loaded": false
}
```

**What a caller actually sees.** A proxied role's `endpoint`/`path` are
unchanged — a client that already discovered them via `GET /capabilities`
keeps POSTing to the same URL it always would; `proxied: true` is purely
informational (a client that ignores it still works). The one visible
difference on the wire is a response header the *raw* OpenAI endpoint
carries — never surfaced in the `/capabilities` JSON itself —
`X-Lobes-Proxied-By: <peer origin>`, present on every answer this box
produced by forwarding, absent on every locally-served answer. See
[`docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in`](gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in)
for the full marker-header and failure-mode contract.

**`ready` is the live peer probe, not a hardcoded claim.** A proxied role's
`ready` is never forced `true` just because a local process happens to be
healthy — a background thread probes the declared peer's own `GET
/v1/models` and `ready` reflects whether the peer actually lists the id this
box would forward to it. A dead or misconfigured peer means `ready: false`
(or the id drops off `/v1/models` entirely) even though `proxied: true` is
still declared — declaring the intent to proxy is not evidence the peer is
reachable right now.

**The honesty invariants this state carries forward, unchanged:**

- **#91 (no silent substitution)** — a proxied `senses` request is answered
  by `senses` running on the peer, never quietly served by a different,
  locally-feasible model. The peer's own served id is what comes back; a
  peer that itself declines the role (`404 role_infeasible`) is relayed
  terminally, naming the peer, never silently retried against something else.
- **#92 (operator-declared origins, never derived)** — `hosted_by` is always
  the literal `<PREFIX>_PEER_ORIGIN` an operator typed into `.env`; nothing
  here infers a peer from hostnames, interfaces, or DNS.
- **Single-hop** — a role proxied on this box is never proxied a second time:
  a request already carrying the internal hop marker that would need to
  depart again is refused rather than forwarded onward, so two
  misconfigured boxes pointing at each other fail fast instead of looping.

**Default off, byte-identical.** With no `<PREFIX>_PEER_PROXY` armed
anywhere, no role in this deployment is ever proxied — every payload here
looks exactly as it did before this state existed (a `feasible: false` role
carries `hosted_by` at most, never `proxied`).

## Serving: `lobes up <role>` and `colleague-stack`

`lobes up` starts (or, with `--down`, stops) **one** role's gear without
touching the rest of the fleet:

```bash
lobes up cortex --apply             # docker compose up -d vllm-primary
lobes up senses --apply             # docker compose up -d vllm-multimodal
lobes up muse --apply               # docker compose up -d vllm-muse (muse-hosting shape only)
lobes up embedder --apply           # docker compose up -d vllm-embed
lobes up reranker --apply           # docker compose up -d vllm-rerank
lobes up stt --apply                # requires the --audio overlay
lobes up tts --apply                # requires the --audio overlay
lobes up colleague-stack --apply    # the SIX default roles at once (requires --audio scaffolded)
```

Dry-run by default (prints the exact `docker compose …` command); `--apply`
commits. `colleague-stack` is a first-class bundle — the four default fleet
roles **plus** the audio overlay's `stt`/`tts` — not a compose `profiles:`
tag, because tagging the already-default-on services with a profile would
demote them out of the default fleet (a regression). If the audio overlay
isn't scaffolded, `colleague-stack` (and `up stt`/`up tts`) fail with a
remediation pointing at `lobes init --fleet --audio --apply`, rather than
silently starting only four of the six roles. **`colleague-stack` stays the
six default-hosted roles — `muse` is deliberately excluded** (its `vllm-muse`
service is compose-profile-gated, so bundling it would break the target on
every non-muse deployment). `lobes up muse` works on a muse-hosting
deployment and errors helpfully — naming the fix — when the deployment's
`COMPOSE_PROFILES` doesn't include `muse`.

## Measurement: `lobes measure`

Read-only, per-role **runtime** metrics — never a task-quality or correctness
claim (lobes measures serving performance; whether an *answer* was good is
Colleague's call):

```bash
lobes measure              # all seven roles, table
lobes measure --json       # all seven roles, JSON
lobes measure --role cortex --json
```

Metrics are grouped by the role's family:

- **LLM roles** (`cortex`, `senses`, `muse`): `ttft_ms`, `decode_tps`, `prefill_tps`,
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
| `cortex+senses` | `cortex` and `senses` side by side (the `machine-as-brain` default duo — a mesh-brain deployment shape can drop one of the two; see `docs/deployment-shapes.md`). |
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
2. Read the role you want (`cortex`, `senses`, `muse`, `embedder`, `reranker`,
   `stt`, `tts`) out of the response — its `endpoint`, `model`, and `path`.
3. `POST <endpoint><path>` with `"model": <model>` and the role-appropriate
   body shape (chat messages for `cortex`/`senses`/`muse`, `input` for
   `embedder`, `query`+`documents` for `reranker`).

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

Because `cortex`, `senses`, and `muse` are **gateway-fronted**, they (along
with `embedder`/`reranker`) share the **same `endpoint`** — the gateway's base
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
- [`docs/deployment-shapes.md`](deployment-shapes.md) — the orthogonal
  deployment-shape axis: which of these seven roles a given box hosts at all,
  the cross-box honest-referral surface for a role it doesn't, and the
  opt-in proxy-lobes extension (the awake/asleep/proxy table, the pairwise
  key contract, a worked example).
- `lobes explain roles` — the in-CLI version of this doc.
- `lobes explain fleet` / `lobes explain gateway` — routing semantics.
- `tests/test_colleague_contract.py` — the end-to-end proof of the client flow
  and the runtime-only boundary described above.
- `tests/test_roles_proxied.py` / `tests/test_gateway_proxy.py` — the proxied
  role state and the data-plane forward it rides on.
