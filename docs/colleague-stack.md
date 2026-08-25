# The Colleague stack: ten roles, one contract

> The ten first-class, Colleague-facing roles lobes exposes over the fleet —
> `cortex` / `senses` / `muse` / `worker` / `hand` / `embedder` / `reranker` /
> `stt` / `tts` — how a caller discovers them, drives them, measures them, and
> the before→after context migration that shipped alongside this contract
> (issue #81; `muse` joined as the seventh, opt-in-hosted role, `worker` as the
> eighth, thor-worker-lobe plan, and `hand` as the ninth, hand-lobe plan).
> **`muse` is currently DORMANT/unhosted mesh-wide** — see the callout below
> the role table.
>
> **Topology swap, deviation d1 (2026-08-20).** `worker` is no longer the
> multimodal `unsloth/Qwen3.6-35B-A3B-NVFP4` on the Thor — the Thor's
> Mamba-2 SSD decode path wedged on this fleet's pinned nightly
> (`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`), so
> operator-approved deviation d1
> (`.devague/deliveries/nemotron-lightning-worker.json`) moved `worker` to
> `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` on the **DGX Spark
> GB10** instead, and moved `cortex` to the **Thor**, serving it locally at
> its full 1M window (MTP off — sm_110 has no GDN-MTP decode kernel on this
> digest). See
> [`nemotron-3.5-lightning-30b-a3b-nvfp4.md`](nemotron-3.5-lightning-30b-a3b-nvfp4.md),
> [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md), and
> `CLAUDE.md`'s "Colleague roles" section for the deployed picture; the
> per-role table and responsibility tokens below are updated for it.

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

## The ten roles

| Role | Backend / service | Endpoint path | What it's for |
|---|---|---|---|
| `cortex` | `primary` (generate, **hosted on the Jetson AGX Thor since d1**) | `POST /v1/chat/completions` | Reasoning, deciding, planning, tool use, repo actions — the final authority. |
| `senses` | `multimodal` (generate) | `POST /v1/chat/completions` | Intake/perception (text+image) and speaking back to the user. Does **not** decide or act. |
| `muse` | `muse` (generate, **opt-in hosting, currently DORMANT/unhosted**) | `POST /v1/chat/completions` | Creative generation, long-form writing, ideation, a divergent second opinion. Proposes; never decides or acts. |
| `worker` | `worker` (generate, **opt-in hosting, Lightning on the DGX Spark since d1**) | `POST /v1/chat/completions` | Fast ground-work execution — bulk transforms, drafting, repo inspection, running authorized commands — **and repo actions**, under `cortex`'s direction. TEXT-ONLY, non-coding (see the d1 callout above). Never the final decision or a security call. |
| `associate` | `associate` (generate, **opt-in hosting**) | `POST /v1/chat/completions` | The same fast ground-work as `worker` — execution, bulk transforms, drafting, repo **inspection**, running authorized commands — but it hands the result BACK instead of enacting it: `repo_action` is **forbidden**. "They do, but not act." |
| `hand` | `hand` (generate, **default-hosted everywhere**) | `POST /v1/chat/completions` | The fine-tuning base and trained specialist — domain mastery via LoRA adapters. Also the `minor`/`cheap` tier and the pressure-policy **servable floor**. Never decides, acts on the repo, or makes a security call. |
| `embedder` | `embed` (pooling) | `POST /v1/embeddings` | Dense text embeddings for memory/retrieval. |
| `reranker` | `rerank` (pooling) | `POST /v1/rerank` (+ `/v1/score`) | Reordering/scoring retrieved candidates. |
| `stt` | Parakeet (audio overlay, opt-in) | `POST /v1/audio/transcriptions` | Speech-to-text. |
| `tts` | Chatterbox (audio overlay, opt-in) | `POST /v1/audio/speech` | Text-to-speech. |

**`cortex` is the final decision authority, not the only lobe that acts.**
Since `worker` joined as the eighth role, two lobes may act on the repo:
`cortex` (unrestricted — `code_repo_actions` plus `final_authority`) and
`worker` (`repo_action` allowed, but forbidden `final_decision` and
`security_decision` — it executes ground work under `cortex`'s direction, it
never decides on its own authority). Every other role — `senses`, `muse`,
`embedder`, `reranker`, `stt`, `tts` — still carries no acting authority at
all — including `associate`, the tenth role, which is `worker` with
`repo_action` moved from the allowed column to the forbidden one. See
"Responsibilities and forbidden responsibilities" below for the
full division of labour.

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

`cortex`, `senses`, `muse`, `worker`, `embedder`, and `reranker` are always
enumerated (present with `loaded=false` if their gear isn't wired in this
deployment — `muse` and `worker` additionally report `feasible=false` unless
their respective hosting shape declares them, see the note below); `stt`/`tts`
require `lobes init --fleet --audio`. **`brain` is not a valid role name** —
`cortex` is the only reasoning/decision-*authority* role (`worker` may act,
per the callout above, but it never decides).

> **`muse` and `worker` are OPT-IN CORE ROLES — machine-as-brain never hosts
> either.** The `nvidia/Gemma-4-31B-IT-NVFP4` checkpoint behind `muse` and the
> `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` checkpoint now behind
> `worker` (since d1, replacing `unsloth/Qwen3.6-35B-A3B-NVFP4`) are both too
> heavy to co-reside with the default `cortex`+`senses` duo on a 128 GB box,
> so the default shape's hosted set stays the SEVEN default-hosted roles
> (`DEFAULT_HOSTED_ROLES` in `lobes/profiles/shapes.py`) while the contract
> set capabilities reports (`COLLEAGUE_ROLES`) is eight. Only an explicit
> hosting shape serves either: `lobes init --shape thor-muse` for `muse` —
> **DECLARED/UNVALIDATED** as of this writing, and now additionally
> **DORMANT** (see below) — a 2026-07-17 live boot measured the budget, but
> the acceptance run/transcript never landed, #108 (see
> [`docs/gemma-4-31b-nvfp4.md`](gemma-4-31b-nvfp4.md)); the `thor-worker`
> shape for `worker` is now **LIVE, but rendered on the Spark card, not the
> Thor** (d1, 2026-08-20) — see
> [`deployment-shapes.md`](deployment-shapes.md#shapes-are-card-agnostic-data-proven-live-by-d1)
> for how a shape named `thor-worker` ended up hosting `worker` on a
> different card; the measured budget lives in
> [`docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md`](nemotron-3.5-lightning-30b-a3b-nvfp4.md).
> On every
> non-hosting deployment both are honestly `feasible: false` (and, uniquely
> among the six other roles, an unwired `muse`/`worker` *defaults* to
> infeasible even on a stale pre-muse/pre-worker `.env` — see
> [`docs/gateway-fleet.md`](gateway-fleet.md#generate-lane-tier-aliases)), so
> `model=muse` / `model=worker` 404s `role_infeasible` — referable and
> proxyable like every core role — rather than silently falling back to
> `cortex`.
>
> **`muse` is currently DORMANT/unhosted mesh-wide.** The physical Jetson AGX
> Thor — the one box that ran (unvalidated) `thor-muse` — moved to hosting
> `worker` instead: an operator decision (thor-worker-lobe plan) that no box
> in the mesh currently hosts the 31B `muse`, and the Thor's deployment
> declares no `MUSE_PEER_ORIGIN`, so `model=muse` 404s `role_infeasible` with
> **no** `hosted_by` referral anywhere. The `muse` role, its catalog entry,
> and the `thor-muse` shape all **stay in-tree** (cite-don't-delete) — dormant,
> not deleted — so this contract still enumerates `muse` with `loaded=false`,
> `feasible=false`, and no `hosted_by`, exactly like any other unhosted opt-in
> core role with no declared peer.

### Responsibilities and forbidden responsibilities

Each role carries a declared division of labour — what it is expected to own,
and (for `senses`, `muse`, and `worker`) what it must **not** do. These are
**runtime-descriptor tokens, not correctness claims** — lobes does not grade
whether a role did its job well; that judgment is Colleague's (see
"Runtime-only, always" below).

| Role | `responsibilities` | `forbidden_responsibilities` |
|---|---|---|
| `cortex` | `reasoning`, `deciding`, `planning`, `tool_use`, `code_repo_actions`, `validation`, `final_authority` | *(none — cortex is the final authority)* |
| `senses` | `intake`, `normalize_input`, `classify_intent`, `prepare_context_packet`, `speak_back` | `final_decision`, `repo_action`, `security_decision` |
| `muse` | `creative_generation`, `long_form_writing`, `ideation`, `style_variation`, `divergent_second_opinion`, `tool_use` | `final_decision`, `repo_action`, `security_decision` — muse proposes, cortex decides |
| `worker` | `execution`, `ground_work`, `bulk_transform`, `drafting`, `repo_inspection`, `run_authorized_commands`, `tool_use`, `repo_action` | `final_decision`, `security_decision`, `code_authoring` — worker acts under cortex's direction, never on its own authority, and does not author code |
| `associate` | `execution`, `ground_work`, `bulk_transform`, `drafting`, `repo_inspection`, `run_authorized_commands`, `tool_use` | `final_decision`, `security_decision`, `code_authoring`, `repo_action` — worker's forbidden list PLUS `repo_action`: associate produces the work, cortex or worker enacts it |
| `embedder` | `vectorization`, `memory_retrieval_input` | *(none)* |
| `reranker` | `retrieval_ordering`, `relevance_refinement` | *(none)* |
| `stt` | `transcribe`, `audio_input_to_text` (+ `realtime_vad_session` when the audio overlay is wired and feasible — see below) | *(none)* |
| `tts` | `speech_output`, `synthesize` | *(none)* |

**`worker` is the first role besides `cortex` permitted `repo_action`.**
Every other non-`cortex` role (`senses`, `muse`, `embedder`, `reranker`,
`stt`, `tts`) forbids it or has no acting authority to begin with. `worker`'s
`repo_action` is deliberately narrower than `cortex`'s: `cortex` also carries
`final_authority` and no forbidden list at all, while `worker`'s forbidden
list still bars `final_decision` and `security_decision` — worker executes
ground work (bulk transforms, drafting, repo inspection, running authorized
commands) UNDER `cortex`'s direction, it never decides on its own authority.
**Since deviation d1 (2026-08-20), `worker` is TEXT-ONLY and
non-coding** — the checkpoint behind it
(`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, hosted on the DGX
Spark) carries no vision tower, unlike the multimodal Qwen checkpoint it
replaced, and `code_authoring` is forbidden outright. See
[`nemotron-3.5-lightning-30b-a3b-nvfp4.md`](nemotron-3.5-lightning-30b-a3b-nvfp4.md)
for the checkpoint facts and
[`qwen3.6-35b-a3b-nvfp4.md`](qwen3.6-35b-a3b-nvfp4.md) for the demoted
multimodal predecessor.

**`stt`'s `realtime_vad_session` responsibility is additive and
honesty-gated (issue #149).** It names the `/v1/realtime` WebSocket
server-side-VAD session surface — one connection streams PCM in and
receives speech-start/speech-stop boundary events plus committed-turn
transcriptions, replacing client-side energy-threshold endpointing. Unlike
the rest of this table, it is **not** a static entry in
`ROLE_RESPONSIBILITIES` — `lobes.roles._resolve_audio_role` appends it to
`stt`'s `responsibilities` tuple only when the audio overlay is actually
wired on this deployment (`AUDIO_URL` configured, i.e. `lobes init --fleet
--audio`) **and** the lane hasn't been declared off (`STT_FEASIBLE=false`).
A text-only fleet (no audio overlay) or a declared-off `stt` lane reports the
base two-token tuple only, on both `lobes capabilities` and `GET
/capabilities` — the same `loaded`/`feasible` honesty discipline every other
field in this contract already follows, not a new signal. See
`tests/test_cli_capabilities.py` and `tests/test_colleague_contract.py` for
the positive (audio-enabled) and negative (text-only) proof.

Three roles carry `tool_use`: `cortex`, `muse`, and `worker`. They are not
equivalent, and the `forbidden_responsibilities` column is what separates
them — `cortex` may act on a tool result with full authority (`repo_action`
via `code_repo_actions`, plus `final_authority`); `worker` may also act
(`repo_action` is present, not forbidden) but never decides
(`final_decision`/`security_decision` stay forbidden) — it executes under
`cortex`'s direction; `muse` may only *research* with a tool call
(`repo_action` is forbidden outright). muse calling `read_file` to ground a
proposal is in-contract; muse calling anything that writes is not. worker
calling a write tool to execute ground work IS in-contract, but worker
choosing to merge, deploy, or otherwise make the final call is not — that
authority stays `cortex`'s alone. `senses` has no `tool_use` at all: it is
intake/perception, even though its Gemma lane *can* serve tool calls (see `tools`,
below — a capability of the lane, not a licence for the role).

### `associate` — the tenth role, the doer that does not act

`associate` serves the **same checkpoint the `worker` seat holds**
(`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`) under a **different
authority**: every doer token worker carries for producing a result, minus
`repo_action`. One gear, two public addresses.

Why a role and not a responsibilities token on `worker`? Because a token
cannot give a **separate public address**, and that is exactly what the
operator wanted: the `worker` seat is being kept free for a possible future
worker/cortex switch, and a caller that wants "do the ground work but change
nothing" needs a name it can address that will not move when that switch
happens. (The behaviour difference alone would indeed have been a token — see
the irreversibility box below, and the `hand` precedent above, which handles
the identical `repo_action` asymmetry inside an existing role.)

Its responsibilities list is deliberately **shorter than worker's**, on the
same `hand` precedent: adding a responsibility later is contract-compatible,
removing one is a break, so the conservative list ships first. worker's
agent-work tokens (`action_selection`, `retrieval_synthesis`,
`summarization`, `log_digestion`, `structured_extraction`) are not claimed
here — not because associate could not serve them, but because nothing
measured says it should own them in the division of labour.

Three facts follow from it being an **opt-in core role**
(`OPT_IN_CORE_ROLES`, alongside `muse` and `worker`):

* `machine-as-brain` NEVER hosts it, `base.toml` vetoes it on an unrecognised
  card, and only an explicit associate-hosting shape serves it;
* an **unwired** associate is INFEASIBLE by default (`OPT_IN_BACKENDS`), so
  `model=associate` 404s `role_infeasible` — referable and proxyable through
  the `ASSOCIATE_PEER_*` channels, never a silent fallback to cortex;
* under pressure it **sheds** (HTTP 429 + `Retry-After`) exactly like
  cortex/senses/worker/muse. It is **not** a servable floor — `hand` remains
  the only one.

A fourth fact is specific to this role: **hosting it requires an inbound
gateway key.** The `vllm-associate` container publishes no host port (the
gateway is the only way in), and the checkpoint's published Jetson recipe —
run verbatim during the 2026-08-25 spike — put an uncredentialed 30B generate
lane on the operator's tailnet that two distinct peers queried unprompted
within seconds. The shipped lane does not inherit that recipe's host
networking, and `lobes doctor` fails (error severity, `associate_auth_gate`)
any deployment that wires `ASSOCIATE_BASE_URL` while setting neither
`GATEWAY_API_KEY` nor `CULTURE_VLLM_API_KEY`. See
`docs/evidence/2026-08-26-associate-gateway-auth-front.txt` and
[`docs/gateway-fleet.md`](gateway-fleet.md)'s auth section.

In the capability ladder it takes the **highest non-cortex rung**:
`hand` < `multimodal` < `worker` < `muse` < `associate` < `main`/`cortex`.
Its role name IS its backend/tier name, like `muse` and `worker`.

### `hand` — the ninth role, the trained specialist

`hand` (LiquidAI `LFM2.5-1.2B-Instruct`) is the fleet's **designated
fine-tuning base**. The metaphor is **muscle memory**: one cheap base, many
LoRA adapters, each mastering a domain.

The distinction from `worker` is the point of having both:

| | `worker` | `hand` |
|---|---|---|
| what it is | an untrained **generalist doer** | a trained **specialist** |
| how it gets good | it is already big (35B-A3B) | someone taught it (a LoRA adapter) |
| breadth | anything, adequately | a few things, extremely well |
| may act on the repo | **yes**, under cortex's direction | no (v1 — see below) |
| hosting | opt-in, one box | **default, every box** |

At ~1.2B parameters (~2.4 GiB bf16) it is cheap enough to co-reside on *every*
card, which is what makes it different in kind from the other generate lobes.
That has three consequences worth stating plainly:

- It is **default-hosted by every built-in shape**, including the mesh-lobe
  shapes that drop a heavy lobe. A caller always has a local generate lane.
- It is **never proxied**. `hand` is deliberately absent from the peer
  origin/proxy/key channels (`NEVER_PROXIED_BACKENDS`): referral exists so a
  box that *cannot* host a lobe can still reach it, and that situation does not
  arise here.
- It is the **servable floor**. Under pressure `cortex`/`senses`/`worker`/`muse`
  all shed with 429; `hand` is served regardless.

It also **replaced `Qwen/Qwen3.5-4B` as the `minor`/`cheap` tier**. Those tier
spellings still work and now resolve to `hand`; the 4B stays in the catalog as
a plain candidate (cite-don't-delete), selectable via `lobes switch`, but no
tier resolves to it.

**Addressing an adapter.** `model=hand` serves the base — it never 404s just
because the inventory is empty. `model=hand:<domain>` serves that adapter. An
*undeclared* `hand:<domain>` is refused with `model_not_found`; it is never
silently downgraded to the base, because a caller who asked for the legal
specialist and got the generalist has been lied to. Adapters are declared once
in `HAND_LORA_MODULES` (read by both the engine and the gateway, so they cannot
disagree), fixed at boot — there is no runtime hot-load.

**Per-card status, updated 2026-08-20.** `hand` (LFM2.5-1.2B) is
default-hosted everywhere, but its inference health is not uniform:
**Orin and Spark are VALIDATED** (known-answer + structured tool-calls PASS
on both — `docs/evidence/2026-08-20-accept-hand-spark.txt` for the Spark
half); **Thor stays BLOCKED**, now **re-attributed** away from its original
filing. #181 was originally filed as a LoRA embedding-slot boot failure;
the 2026-08-20 re-run
(`docs/evidence/2026-08-20-hand-thor-blocked-reattributed.txt`) found boot
and LoRA allocation now succeed on both the fleet's new `8bd082` nightly
and the old 0.23.1 production engine, but **inference itself is corrupt on
the Thor on both engines** — wrong deterministic generations, escalating to
a CUDA unspecified-launch-failure / EngineCore death within 1–3 requests.
The identical config passed cleanly on the Spark the same day. This is the
same pattern as the Thor's other sm_110 non-dense-decode gaps that same day
(Lightning's Mamba-2 SSD wedge, the Qwen worker/cortex GDN-MTP kernel gap —
see [`docs/machine-profiles.md`](machine-profiles.md)): dense-transformer
paths serve fine on this digest, non-standard (conv/Mamba/GDN) decode paths
do not. `hand` stays UNSERVED on the Thor (`feasible=false`); #181 should be
re-titled to this inference-level attribution.

**v1 ships zero adapters**, with `--enable-lora` armed and the inventory empty.
The serving half of muscle memory is here; the training half is `unsloth-cli`,
out of tree (`agentculture/unsloth-cli#16`). lobes **serves** adapters and
never trains them — nothing under `lobes/` imports or shells out to unsloth.

`repo_action` is **forbidden** for v1 even though `worker` has it. That is a
deliberate asymmetry, not an oversight: granting it later is
contract-compatible, revoking it is a break, so the conservative list ships
first and `agentculture/lobes-cli#180` tracks granting it once adapters exist.

> ### Adding a role is effectively irreversible
>
> Every name in `lobes.roles.ROLES` becomes a public address: a key on
> `GET /capabilities` and `lobes capabilities`, a `model=` alias, a
> `lobes up <role>` target, a `<PREFIX>_*` env vocabulary, an entry in six
> per-role tables, a row in every card profile and every deployment shape, and
> a line in 28 golden `.env` files. **Removing one later breaks every caller
> that learned to use it** — and by the honesty rule (#92) you cannot soften
> the break by half-serving it.
>
> `hand` was worth that cost because it is a *kind* of lobe the fleet did not
> have: cheap enough to be everywhere, and the only one meant to be taught. A
> tenth role should have to clear the same bar.
>
> **The tenth, `associate`, landed 2026-08-25** (lightning-on-orin plan, t6),
> and it did NOT clear that bar on behaviour alone — by this box's own rule,
> "worker minus `repo_action`" is a responsibilities-token shape, which is how
> `hand` handles the identical asymmetry. What justified it is the one thing a
> token cannot provide: a **separate public address**, so the `worker` seat
> stays free for a possible future worker/cortex switch. That is an operator
> decision about naming, recorded here as such rather than dressed up as a
> capability argument. An eleventh should expect the same scrutiny. If what you want is a different
> checkpoint, that is a catalog change; if it is a different budget, that is a
> profile or shape change; if it is a different behaviour on an existing lane,
> that is a responsibilities token. Reach for a new role only when none of
> those can express it.

## cortex/senses ↔ primary/multimodal — one mapping, three vocabularies

`cortex` and `senses` are **new names layered on the existing `primary` /
`multimodal` backends and the `main` / `multimodal` capability tiers** — no
internal service, container, or env var was renamed. All three vocabularies
resolve to the same backend (`lobes/catalog.py`'s `TIER_ROLE`):

| Backend (`role_hint`) | Primary tier alias | Back-compat tier alias | Colleague role name |
|---|---|---|---|
| `primary` | `main` | `hard` | `cortex` |
| `multimodal` | `multimodal` | `normal` | `senses` |
| `worker` | `worker` | *(none — new with the role)* | `worker` — like `muse`, its name IS the backend/tier name; capability order is `minor` < `multimodal` < `worker` < `muse` < `primary` |
| `muse` | `muse` | *(none — new with the role)* | `muse` — the first role whose name IS the backend/tier name; capability order is `minor` < `multimodal` < `worker` < `muse` < `primary` |
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
lobes capabilities              # human-readable table, all ten roles
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
    "role": str,                          # "cortex" | "senses" | "muse" | "worker" | "embedder" | "reranker" | "stt" | "tts"
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
    "loaded": bool,                       # is this role's backend wired in THIS deployment? (LOCAL wiring only — see below)
    "feasible": bool,                     # can THIS MACHINE serve this role at all? (deployment-shapes)
    "hosted_by": str,                     # OPTIONAL — present only when feasible=false and a peer origin is declared
    "proxied": bool,                      # OPTIONAL — present (and true) only when this box also forwards to that peer
    "replicas": [                         # OPTIONAL, ADDITIVE (issue #199) — present only when a *_PEER_ORIGINS pool is declared
      {
        "origin": str,                    # "local" for this box's own replica, else the peer origin
        "local": bool,
        "ready": bool,
        "busy": bool,
        "running": int,
        "waiting": int,
        "compatible": bool,               # served id + quantization + max context + runtime all agree
        "reason": str,                    # e.g. "" when compatible, else which field differs
        "fingerprint": {...} | None,
      },
      ...
    ],
    "fingerprint": {...} | None           # OPTIONAL, ADDITIVE (issue #199) — this box's OWN live-probed lane fingerprint
  },
  ...
}
```

`feasible` is always present (a hardware/deployment fact — see
[`docs/deployment-shapes.md`](deployment-shapes.md)); `hosted_by` and
`proxied` are **optional keys**, added only for a role this box does not
host, and only when the operator declared a peer for it — see
[A third role state: proxied](#a-third-role-state-proxied) below for the full
three-state contract. `replicas` and `fingerprint` are a SEPARATE, purely
**additive** extension (issue #199, the cortex replica pool): every existing
key here — `feasible`, `hosted_by`, `proxied`, `ready`, `loaded` — keeps its
documented type and single-owner meaning even on a pooled role; a payload
built with no `*_PEER_ORIGINS` declared anywhere has no `replicas` key at all
and is byte-identical to the pre-pool contract. The pool composes on top of
the awake/proxied states above, not a fourth state, and its full mechanism —
selection policy, marker headers (`X-Lobes-Served-By` alongside the existing
`X-Lobes-Proxied-By`, plus `X-Lobes-Route-Reason` on every pooled answer),
the `<PREFIX>_PEER_ORIGINS`/`_PEER_API_KEYS` config family, and the
failure/rollback table — lives in
[`docs/gateway-fleet.md#replica-pools-one-lobe-n-replicas-opt-in-cortex-validated-only`](gateway-fleet.md#replica-pools-one-lobe-n-replicas-opt-in-cortex-validated-only).
**Status: VALIDATED live 2026-08-25 (#108) — cortex only**, on the Spark+Thor
NVFP4 pair (`docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`); the CLI view is `lobes capabilities --replicas` / `lobes endpoint
<role> --replicas`, which also prints a "would choose: `<origin>`
(`<reason>`)" line.

**`tools`** answers "can I put an OpenAI `tools` array on a request to this
role?" — `true` for the four generate lobes (`cortex`/`senses`/`muse`/`worker`), `false`
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
it directly (issue #87). All ten roles (`cortex`/`senses`/`muse`/`worker`/`associate`/`hand`/
`embedder`/`reranker` **and** `stt`/`tts`) report the same base URL because routing happens via the
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
consumable. The gateway-fronted roles (`cortex`/`senses`/`muse`/`worker`/
`embedder`/`reranker`) still report `ready` as a same-cost-as-`loaded` boolean
**unless the role is proxied** (below), in which case `ready` reflects a live
probe of the *peer*, not a local boolean. No `ready` value is a task-quality
claim.

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
client can always render all ten roles. (An unwired `muse` or `worker`
additionally defaults to `feasible: false` — the opt-in-hosting honesty rule
above.)

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

**`loaded` is a LOCAL wiring fact, and it does not tell you whether a proxied
role works.** It answers "does a backend for this role exist in *this box's*
routing table" — never "is this role usable". For a proxied role the usable
signal is `ready` (which for a proxied role is the *peer* probe's verdict:
the peer answered and its own `/v1/models` lists exactly this id). Two roles
in the same proxied state, both forwarding happily, can therefore report
different `loaded` values:

- `loaded: false` is the common case — a dropped role realistically has no
  `<PREFIX>_BASE_URL`, so no local `Backend` is built (the example above).
- `loaded: true` also occurs, and is **not** a bug in the payload: any
  `<PREFIX>_BASE_URL` still present wires a local `Backend` even when no such
  container runs here. Note `multimodal` is the one optional generate backend
  whose fleet-compose default is **non-empty**
  (`MULTIMODAL_BASE_URL=${MULTIMODAL_BASE_URL:-http://vllm-multimodal:8000}`,
  where `MINOR_BASE_URL` and `MUSE_BASE_URL` both default to empty), and
  because `${VAR:-default}` ignores an explicit empty value too, a proxied
  `senses` reports `loaded: true` on every fleet deployment and **cannot** be
  unwired from `.env`.

Because of that, `lobes capabilities` does not render `loaded` verbatim: a
role this box serves by forwarding shows **`by-proxy`** in the `loaded`
column (derived from `feasible: false` + `proxied: true`), so the table reads
the same for every proxied role regardless of leftover local wiring. That is
a **CLI display state only** — this JSON contract is unchanged, `loaded`
stays a `bool`, and a programmatic consumer branching on it is unaffected.

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
lobes up muse --apply               # docker compose up -d vllm-muse (muse-hosting shape only — currently no box hosts one, see the dormant callout above)
lobes up worker --apply             # docker compose up -d vllm-worker (verb wired; live on the Spark since d1, errors helpfully off any other non-hosting box)
lobes up embedder --apply           # docker compose up -d vllm-embed
lobes up reranker --apply           # docker compose up -d vllm-rerank
lobes up stt --apply                # requires the --audio overlay
lobes up tts --apply                # requires the --audio overlay
lobes up colleague-stack --apply    # the SEVEN default roles at once (requires --audio scaffolded)
```

Dry-run by default (prints the exact `docker compose …` command); `--apply`
commits. `colleague-stack` is a first-class bundle — the four default fleet
roles **plus** the audio overlay's `stt`/`tts` — not a compose `profiles:`
tag, because tagging the already-default-on services with a profile would
demote them out of the default fleet (a regression). If the audio overlay
isn't scaffolded, `colleague-stack` (and `up stt`/`up tts`) fail with a
remediation pointing at `lobes init --fleet --audio --apply`, rather than
silently starting only some of them. **`colleague-stack` stays the
seven default-hosted roles — `muse` and `worker` are deliberately excluded,
while `hand` IS included (default-hosted, no compose-profile gate)**
(their services are compose-profile-gated, so bundling either would break the
target on every non-hosting deployment). `lobes up muse` works on a
muse-hosting deployment and errors helpfully — naming the fix — when the
deployment's `COMPOSE_PROFILES` doesn't include `muse`; `lobes up worker`
mirrors that exact mechanic and is live on the Spark (deviation d1,
2026-08-20) — the `vllm-worker` compose service and the worker-hosting shape
both landed, just on a different card than the plan first targeted.

## Measurement: `lobes measure`

Read-only, per-role **runtime** metrics — never a task-quality or correctness
claim (lobes measures serving performance; whether an *answer* was good is
Colleague's call):

```bash
lobes measure              # all ten roles, table
lobes measure --json       # all ten roles, JSON
lobes measure --role cortex --json
```

Metrics are grouped by the role's family:

- **LLM roles** (`cortex`, `senses`, `muse`, `worker`): `ttft_ms`, `decode_tps`, `prefill_tps`,
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
2. Read the role you want (`cortex`, `senses`, `muse`, `worker`, `embedder`,
   `reranker`, `stt`, `tts`) out of the response — its `endpoint`, `model`,
   and `path`.
3. `POST <endpoint><path>` with `"model": <model>` and the role-appropriate
   body shape (chat messages for `cortex`/`senses`/`muse`/`worker`, `input`
   for `embedder`, `query`+`documents` for `reranker`).

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

Because `cortex`, `senses`, `muse`, and `worker` are **gateway-fronted**, they
(along with `embedder`/`reranker`) share the **same `endpoint`** — the
gateway's base URL — and routing between them happens purely via the `model`
field the contract handed back. `stt`/`tts` resolve to the audio-overlay
bridge URL instead.

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
  deployment-shape axis: which of these ten roles a given box hosts at all,
  the cross-box honest-referral surface for a role it doesn't, and the
  opt-in proxy-lobes extension (the awake/asleep/proxy table, the pairwise
  key contract, a worked example).
- `lobes explain roles` — the in-CLI version of this doc.
- `lobes explain fleet` / `lobes explain gateway` — routing semantics.
- `tests/test_colleague_contract.py` — the end-to-end proof of the client flow
  and the runtime-only boundary described above.
- `tests/test_roles_proxied.py` / `tests/test_gateway_proxy.py` — the proxied
  role state and the data-plane forward it rides on.
