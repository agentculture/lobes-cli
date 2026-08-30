# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`lobes` is the tooling that **runs, assesses, and switches** the local,
OpenAI-compatible vLLM model the Culture mesh consumes. The binary is **`lobes`**
(`lobes switch`, `lobes assess`, `lobes serve`, …; `model` is a deprecated alias).

**`lobes` is one identity — the tool *and* the deployed agent:**

- **lobes** is the *repo* and the *tool*. It is a normal CLI/PyPI sibling
  (Python package `lobes`, binary `lobes`, distributed as `lobes-cli`).
- **lobes** is *also* the *agent* deployed *on* the model it serves.
  `AGENTS.md` + `culture.yaml` are that agent's runtime identity (the `acp`
  system prompt and the `suffix: lobes` / `backend: acp` / `model:
  vllm-local/...` declaration). Same name, one identity: the gear runs the model
  and the agent rides on it. (It used to be a separate agent, `lepenseur`; that
  name is retired.)

The served model is **`vllm-local/unsloth/Qwen3.8-27B-NVFP4`** (a Qwen3.5-arch
27B with hybrid Mamba/linear-attention layers; **MULTIMODAL** — image and video
intake through its own ViT; a **self-hosted** MTP draft head baked into the
checkpoint, so vLLM speculative decoding (Multi-Token Prediction) works with no
external draft repo; compressed-tensors NVFP4, 262144 (256K) native; thinking
mode with a reasoning trace). It replaced the previous primary,
`unsloth/Qwen3.6-27B-NVFP4` (now a demoted candidate, kept per
cite-don't-delete), 2026-08-19 — same architecture family, so the swap is a
checkpoint change within the same engine-support family, not a new-arch
bring-up. This is the **`cortex`** role — the fleet's
reasoning/deciding/final-authority lobe (issue #81), and since the 2026-07-31
promotion it is **the first role that can both see an image and decide**: the
role contract forbids `senses` from `final_decision`/`repo_action` and `worker`
from `final_decision`/`security_decision`, so before this every visual decision
had to be handed to a role barred from making it.

**Served context depends on deployment shape:** the legacy single-model scaffold
(`lobes serve`, no fleet) serves the full 256K solo; the **spark-lobe** shape
(what the DGX Spark runs — `senses` dropped to a peer) serves cortex at its
**full native 262144 (256K)** window with `gpu_mem_util=0.58`, drafting with
**DSpark** (`RadixArk/Qwen3.8-27B-DSpark`, block 7, revision pinned) rather
than the checkpoint's own MTP head — the **d4 adoption, 2026-08-25**
(`docs/dspark-speculation.md`; measured 2026-08-24,
`docs/evidence/2026-08-24-spike-dspark-cortex-spark.txt`; the shape's render
proved against the live container in
`docs/evidence/2026-08-25-accept-spark-lobe-dspark-render.txt`). DSpark beats
MTP-n2 on code (46.20 vs 24.69 tok/s) and reasoning and **loses on prose**
(13.71 vs 16.65) — a named cost of this default. Note there is **no ergonomic
per-box override** for it today: a shape override beats the card profile and a
re-render force-writes the key, so changing it means selecting a different
shape or forking the shape file (issue #204). Measured KV pool at the adopted pair:
760,806 tokens = **2.90× concurrency at 262144**.

That shape previously declared a YaRN-extended **1M window**
(`max_model_len=1048576`), MEASURED live 2026-08-19
(`docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt`): same `gpu_mem_util=0.58`
(the 0.60 hypothesis was refused twice at boot; 0.58 booted after the opt-in
embed-deep 4B gear was stopped to fund the budget — the operator's reclaim
decision), KV pool 42.07 GiB = 1,271,476 tokens = **1.21× ceiling at full 1M**
(arithmetic, effectively single-request at max depth). A 328K-token needle
retrieval — beyond the 262144 native ceiling — passed live, and an 8-prompt QA
comparison measured ZERO quality cost from always-on YaRN (7/8 native vs 7/8
YaRN, identical failure). **That window is WITHDRAWN, not disproven:** the
1.36B DSpark drafter's KV cost does not fit beside it at util 0.58 (vLLM
refused the boot outright), so adopting DSpark meant trading the window back
down to native. The YaRN `hf_overrides` block itself is **kept** — every
DSpark arm was measured with it in force — while `allow_long_max_model_len`
was dropped as inert at exactly the native ceiling. The 1M recipe and the
older MEASURED 2026-07-31 pair (`gpu_mem_util=0.44` / `max_model_len=262144`,
no YaRN) are both kept in the shape TOML as documented rollbacks. The machine-as-brain
**fleet duo** declares **128K** (`PRIMARY_MAX_MODEL_LEN=131072`) so cortex can
co-reside with a local multimodal gear — that duo budget is **inherited from the
previous text-only checkpoint and has not been booted with a ViT** (see
`lobes/profiles/builtin/spark.toml`). See `docs/colleague-stack.md#migration-before--after`
and `lobes/profiles/builtin_shapes/spark-lobe.toml` — whose d4 comment block
carries the full rationale for the adopted pair and both rollbacks.

**Deviation d1 (2026-08-20) moved `cortex` off the Spark, onto the Jetson
AGX Thor, locally.** The `spark-lobe`-on-Spark measurements above are now
history, not the deployed reality: a Lightning-worker rollout hit a Thor
NO-GO (`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`) and the
approved topology swap put `cortex` on the Thor instead, freeing the Spark
for `worker` + `hand` + fine-tuning headroom. The Thor now serves
`unsloth/Qwen3.8-27B-NVFP4` **locally**, at the full **1,048,576-token
(1M) YaRN window**, `gpu_mem_util=0.58`, KV pool 1,114,504 tokens, **MTP
OFF** (this checkpoint's GDN decode carries an MTP variant with no sm_110
kernel image on the fleet's current nightly — plain non-MTP GDN decode
works fine), measured **12.1 tok/s** single-stream decode, TTFT ~300 ms —
see `docs/evidence/2026-08-20-accept-cortex-local-thor.txt`. See
`docs/qwen3.8-27b-nvfp4.md` and the `worker`/`muse` paragraphs
below for the rest of the d1 topology.
lobes runs it; the `acp` `vllm-local` provider connects the lobes agent to it.

Three 27B checkpoints remain as **candidates**, kept not deleted
(cite-don't-delete):

- **`unsloth/Qwen3.6-27B-NVFP4`** — the previous default primary, demoted
  2026-08-19 by the Qwen3.8 upgrade. Multimodal (hybrid Mamba/linear-attn +
  ViT, 256K native); every pre-2026-08-19 evidence transcript was measured
  against it.
- **`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`** — an earlier default primary,
  demoted 2026-07-31. Its export dropped the ViT (hence its `--language-model-only`
  and `--tokenizer=mmangkad/…` flags, both now gone from the lane), so it is the
  remaining **text-only** 27B — the pick for a deployment wanting a smaller weight
  footprint and no vision.
- **`mmangkad/Qwen3.6-27B-NVFP4`** — the archived original primary. It used to be
  justified as the tokenizer source the MTP primary served with *and* the only
  vision-capable 27B; **both rationales are now obsolete** — the promoted primary
  ships its own tokenizer and its own ViT. It is kept as a plain candidate.

The `nvidia/Qwen3-32B-NVFP4` dense model also remains a supported candidate — see
`docs/qwen3-32b-nvfp4.md` and `lobes overview --list`.

> **Swapping the served checkpoint breaks every consumer that pins a raw model
> id** — and as of the 2026-07-31 audit, *none* of them address the fleet by role
> name. Read `docs/model-switch-playbook.md` before the next swap; it records the
> ordering that matters (benchmark the incumbent first — that baseline is
> unrecoverable afterwards) and two measurement traps that produced wrong answers
> on this one.

**Thinking continuity — `preserve_thinking` (issue #93).** The cortex/main
vLLM service adds `--default-chat-template-kwargs
'{"preserve_thinking": true}'` next to `--reasoning-parser=qwen3`, so the
served Qwen3.6 chat template retains **all** historical `<think>` blocks
across a multi-turn conversation by default (the template otherwise keeps
only the reasoning after the last user turn). It is default-on but
per-request overridable — a caller's own `chat_template_kwargs` wins over the
server default, so `lobes route`'s terse routing path still forces
`enable_thinking=false` and gets a thinking-free reply. Scoped to the
cortex/main generate lane only — the embed/rerank/senses lanes are untouched.
A read-only preserve-thinking diagnostic (a two-turn prompt-token-count
delta) proves the input-side round-trip is live; the continuity benefit to
output quality is expected and opt-in, not guaranteed by the diagnostic. See
`docs/qwen3.6-27b-text-nvfp4-mtp.md` for the flag and diagnostic detail.

**Strict tool calling with thinking (colleague#320).** Unconstrained
thinking-mode generation can drift off the `qwen3_coder` tool-call template
and get "salvaged" by vLLM's parser into a mangled call; `strict: true` on a
tool schema arms xgrammar structural-tag constrained decoding to make that
impossible, but the served build hardcodes `reasoning=False` at its
structural-tag call site, which breaks the grammar for a thinking model
(`</think>` rejected → 500). The fix is the `qwen3_coder_thinking`
tool-parser plugin (`lobes/vllm_plugins/`, loaded via vLLM's own
`--tool-parser-plugin` file-path surface, cortex/main lane only — mirrors
the `preserve_thinking` #93 scoping) — it derives the grammar's `reasoning`
flag from the request's own `enable_thinking` instead of the hardcoded
`False`. A separate, default-off gateway knob (`GATEWAY_FORCE_STRICT_TOOLS`)
opts existing callers into strict schemas without a client-side change, with
a retry-without-strict fallback on a grammar-compile failure. See
`docs/qwen3.6-27b-text-nvfp4-mtp.md`, `docs/openai-api.md`, and
`docs/gateway-fleet.md` for the mechanism, scope, and knob detail.

**Gemma 4 tool calling — the `gemma4` parser PAIR (2026-07-17).** All three
Gemma 4 lanes (`senses`, the opt-in coder candidate, and `muse`) serve tool
calls with a **matched pair**: **`--tool-call-parser=gemma4`**
(`Gemma4EngineToolParser`) **plus `--reasoning-parser=gemma4`**
(`Gemma4ParserReasoningAdapter`) — mirroring how the cortex lane has always
paired `--reasoning-parser=qwen3` with its `qwen3_coder` tool parser. lobes
previously wired **neither** half, and both failures were silent:

- *Wrong tool parser.* Gemma 4 emits `<|tool_call>call:name{...}<tool_call|>`,
  whose delimiters are **special tokens**. The generic `pythonic` parser runs
  with `skip_special_tokens=True`, never sees them, matches nothing, and vLLM
  relays the model's well-formed call as ordinary assistant **content** with
  `tool_calls: null` / `finish_reason: "stop"` — callers get prose shaped like
  a tool call and no callable one. `pythonic` was a never-validated guess (its
  own `_parser.py` comment flagged it "risk r2, pending live validation"); the
  check finally ran on the live 31B and disproved it.
- *Missing reasoning parser.* The tool parser forces
  `skip_special_tokens=False` (that is how it sees `<|tool_call>`), which also
  exposes Gemma's `<|channel>thought` markers — so the tool parser **alone**
  leaks them into `content`. The reasoning parser is what consumes them.
  Enable both or neither.

**Validated on the 31B `muse` lane only** (physical Thor,
`docs/evidence/2026-07-17-accept-muse-tool-calling-thor.txt`); the 12B lanes
inherit the family rule and are UNVALIDATED (#108). Note
`GATEWAY_FORCE_STRICT_TOOLS` deliberately does **not** arm the muse lane —
measured live, `strict` never engages xgrammar there at all (a schema xgrammar
cannot compile is still served 200), so arming it would advertise a
grammar-constrained lane that isn't one. See
`docs/gemma-4-31b-nvfp4.md#tool-calling`, which also records two rationales for
that exclusion that turned out to be **wrong** (the `supports_required_and_named`
flag, which cortex's own parser shares; and an EngineCore-crash risk that did
not reproduce).

### Colleague roles: cortex / senses / muse / worker / associate / hand / embedder / reranker / stt / tts

Beyond `cortex`, the **fleet** exposes TEN first-class, Colleague-facing
**roles** (issue #81; `worker` joined as the eighth — thor-worker-lobe plan —
and `hand` as the ninth — hand-lobe plan) — the primary contract callers
should address, not raw model ids: `cortex`
(the 27B primary — reasoning/deciding/final authority), `senses` (the Gemma 4
12B multimodal gear — vision intake/perception; never decides or takes repo
actions; the checkpoint declares audio support but it is **not currently
served** on this vLLM path — issue #101 — so `senses` is vision-only in
practice, and the purpose-built `stt` role, below, is the supported path for
speech), `muse` (the opt-in-hosted creative/ideation lobe — **currently
DORMANT/unhosted mesh-wide**, see the paragraph below), `worker` (the
opt-in-hosted fast ground-work DOER, and the first non-`cortex` role allowed
to act on the repo — see the paragraph below), `hand` (the 1.2B LFM2.5
fine-tuning base and trained specialist — default-hosted on EVERY card, the
`minor`/`cheap` tier, and the pressure-policy servable floor; see the
paragraph below), `embedder`
(`Qwen/Qwen3-Embedding-0.6B` → `POST /v1/embeddings`), `reranker`
(`Qwen/Qwen3-Reranker-0.6B` → `POST /v1/rerank` + `/v1/score`), and the
opt-in audio overlay's `stt`/`tts`. Roles are routed by **task family**
(`generate` / `embed` / `score` / `rerank`) and discoverable via `lobes
capabilities` / `lobes endpoint <role>` / gateway `GET /capabilities` — a
JSON contract keyed by role (model / runtime / endpoint / path / context /
quant / mtp / responsibilities / forbidden_responsibilities / ready /
loaded); see `docs/colleague-stack.md` for the full contract.
`cortex`/`senses`/`embedder`/`reranker` are default-on and co-reside on the
DGX Spark GB10: `cortex` serves its **full 128K native context at util 0.30**,
`senses` is trimmed to **32K at util 0.14**, and the two ~0.6B pooling gears
run at `*_GPU_MEM_UTIL=0.06` each — default budget `0.30 + 0.14 + 0.06 + 0.06 =
0.56` on the 128 GB GB10. These are the **machine-as-brain** (default)
values — one box hosting every role it can serve; a mesh-brain **deployment
shape** (below) drops one heavy lobe to a peer box and reclaims its budget
instead of merely co-residing it. `hand` is default-hosted too, at a per-card
util (0.06 on the 128 GB Spark/Thor, 0.10 on the 64 GB Orin). The legacy 4B
`vllm-minor` gear (`COMPOSE_PROFILES=minor`, util 0.10) and the legacy 14B Qwen
(`COMPOSE_PROFILES=middle`, util 0.12) are **opt-in** gears and are not
first-class Colleague roles — and since `hand` took over the `minor`/`cheap`
tier, the 4B is addressable only by explicit model id, exactly like the 14B.
Callers address the generate lane by **capability-tier alias** —
`model=main|hand|multimodal|worker|muse` (back-compat: `hard|minor|cheap|normal`;
capability order `hand` < `multimodal` < `worker` < `muse` < `main`), or the
Colleague-role names `model=cortex|senses` layered on top of
`main`/`multimodal` (`hand`'s, `muse`'s and `worker`'s role names ARE their
tier/backend names); `minor`/`cheap` map to `hand`, and `normal`/`multimodal`
to the Gemma gear, not the demoted 14B. A `hand` LoRA adapter is addressed as
`model=hand:<domain>`. A swap/iowait **pressure policy** SHEDS full-tier `cortex`,
`senses`, `worker`, and `muse` requests with **HTTP 429 + `Retry-After`**
rather than substituting a different model (swap > 75 % or iowait > 50 % →
busy — the former degrade-to-`minor` substitution path was removed outright,
so there is no cheaper-rung fallback for any of the four); an explicit
`hand` request (or its `minor`/`cheap` spellings) is the servable floor and is
always served regardless of pressure. `lobes status --pressure` shows the current busy/warm state.
Start/stop one role at a time with `lobes up <role>` — which since issue #222
actually isolates: every `up` carries `--no-deps` and the base compose gives
the gateway no `depends_on` at all, because `docker compose up -d <service>`
walks that list and a gateway-only restart was therefore recreating every heavy
lane (measured live on the Thor 2026-08-28, including starting a lane the box
declares `MULTIMODAL_FEASIBLE=false`). `gateway` is now an `up` TARGET too (not
a role): `lobes up gateway --build --apply` re-images the front at a new
`MODEL_GEAR_VERSION` without touching the lobes behind it. Or the
seven-default-role bundle, `lobes up colleague-stack` — `muse` and `worker` are
deliberately excluded from the bundle, both being opt-in-hosted, while `hand`
IS included, being default-hosted and un-gated; `lobes up muse` works on
a muse-hosting deployment and errors helpfully when `COMPOSE_PROFILES`
doesn't include `muse`, and `lobes up worker` is landing alongside the
worker-hosting shape, below, to mirror that exact mechanic); measure
per-role runtime with `lobes measure` (hand, muse and worker ride the llm
family)
and compare fleet profiles with `lobes benchmark --profile {cortex-only,
cortex+senses,senses-direct,qwen-nvfp4-vs-bf16,all}`. LoRA adapter training
targets the 4B bf16 `minor` only — the 14B NVFP4 is inference-only, and there
is no `lobes train` verb. See `docs/qwen3-embedding-0.6b.md`,
`docs/qwen3-reranker-0.6b.md`, `docs/gemma-4-12b-nvfp4.md`,
`docs/gemma-4-31b-nvfp4.md`, `docs/qwen3.6-35b-a3b-nvfp4.md`,
`docs/lfm2.5-1.2b-hand.md`,
`docs/gateway-fleet.md`, and `docs/colleague-stack.md` (the ten-role
contract).

**`muse` — the seventh role, currently DORMANT/unhosted mesh-wide.**
Checkpoint: `nvidia/Gemma-4-31B-IT-NVFP4` (Gemma 4 31B IT, NVIDIA's official
modelopt NVFP4 export; 256K native; plain-gemma4 line, **`gemma4` tool
parser**; native MTP declared via the `google/gemma-4-31B-it-assistant`
draft — UNMEASURED on this target). Responsibilities: creative generation,
long-form writing, ideation, style variation, a divergent second opinion,
**`tool_use`** — muse proposes, `cortex` decides (forbidden: final_decision /
repo_action / security_decision, so muse calls tools to RESEARCH a proposal,
never to enact one). Alias `model=muse`. It is an **opt-in core role**
(`OPT_IN_CORE_ROLES`): machine-as-brain NEVER hosts it — a 31B cannot
co-reside with the cortex+senses duo on a 128 GB box — so only an explicit
muse-hosting shape (`thor-muse`, below) serves it; every non-hosting shape
renders nothing for muse — the card's own declaration passes through, so
machine-as-brain stays a byte-identical no-op and only `base.toml`'s veto
emits `MUSE_FEASIBLE=false` — and on a stale/pre-muse `.env` an unwired
`muse` defaults to infeasible (`OPT_IN_BACKENDS` — `model=muse` 404s
`role_infeasible`, referable/proxyable, never a silent fallback to cortex).
Under pressure `muse` sheds (429) exactly like cortex/senses/worker.
**Operator decision (thor-worker-lobe plan, then deviation d1): no box in
the mesh currently hosts `muse`.** The Jetson AGX Thor, the one box that ran
`thor-muse` (DECLARED/UNVALIDATED: the 2026-07-17 live boot measured the
budget, util 0.55 at the full 262144 window, but the acceptance transcript
never landed, issue #108), first moved to hosting `worker` (thor-worker-lobe
plan), then — deviation d1, 2026-08-20, after a Lightning-on-Thor NO-GO
(`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`) — moved again to
hosting `cortex` locally instead, with `worker` relocating to the DGX Spark
(see the `worker` paragraph below). The deployed Thor still declares no
`MUSE_PEER_ORIGIN`, so `model=muse` still 404s `role_infeasible` with **no**
`hosted_by` referral anywhere in the mesh. The `muse` role, its catalog
entry, and the `thor-muse` shape all **stay in-tree** (cite-don't-delete) —
dormant, not deleted — and the tier vocabulary above still ranks `worker` <
`muse`. See `docs/gemma-4-31b-nvfp4.md`.

**`worker` — the eighth role (opt-in hosting), the fast ground-work DOER —
RE-CHECKPOINTED to Lightning on the Spark, deviation d1 (2026-08-20).**
Checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
(`NemotronHForCausalLM`, a Mamba-2/MoE/attention hybrid with ~3B active of
30B total parameters, 1,048,576-token native ceiling, modelopt NVFP4;
served at a trimmed 65536 window). **TEXT-ONLY, non-coding** — this
checkpoint carries no `vision_config`, so `worker` LOST
`image_understanding`/`video_understanding` on this swap (the "seeing
doer" framing below is now dead language, kept only as history).
Responsibilities: execution, ground_work, bulk_transform, drafting,
repo_inspection, run_authorized_commands, `tool_use`, and **`repo_action`**
— worker is the FIRST role besides `cortex` permitted to act on the repo,
under `cortex`'s direction (forbidden: final_decision, security_decision,
**code_authoring** — worker never makes the final call, a security
decision, or writes code, on its own authority). Alias `model=worker`. It
is the **second opt-in core role** (`OPT_IN_CORE_ROLES = ("muse",
"worker")`): machine-as-brain NEVER hosts it, the gateway wires its backend
only behind `WORKER_BASE_URL`, and an unwired `worker` defaults to
infeasible (`OPT_IN_BACKENDS` — `model=worker` 404s `role_infeasible`,
never a silent fallback), mirroring `muse`'s mechanics exactly
(`WORKER_FEASIBLE` / `WORKER_PEER_ORIGIN` / `WORKER_PEER_PROXY` /
`WORKER_PEER_API_KEY`; `base.toml` vetoes it on an unrecognised card just
like `muse`). Under pressure `worker` sheds (429) exactly like
cortex/senses/muse.

**Now hosted on the DGX Spark GB10, not the Thor.** The `thor-worker`
deployment shape's data rendered on the **Spark** card (`lobes init --shape
thor-worker --apply --force` on `spark-f8a9`) after a same-day Thor NO-GO —
Lightning's Mamba-2 SSD decode path wedges indefinitely on this fleet's
pinned nightly on sm_110
(`docs/evidence/2026-08-20-spike-lightning-thor-no-go.txt`; a Thor-specific
recipe on the newer release image `v0.27.1` is unexplored, see that
transcript's addendum). Measured on the Spark, 2026-08-20
(`docs/evidence/2026-08-20-accept-worker-hand-spark.txt`):
`WORKER_GPU_MEM_UTIL=0.30`, `WORKER_MAX_MODEL_LEN=65536` (a progressive
start; 1M is a ceiling, not yet exercised), weights 17.85 GiB loaded, KV
pool 3,560,789 tokens (54.33× concurrency at 65K), `fp8_e4m3` KV cache,
**75.1 tok/s** decode single-stream (no MTP), tool calls PASS via the
`nemotron_v3` reasoning parser + `qwen3_coder` tool parser (the NVIDIA
recipe's pairing, validated live). TTFT medians 75 ms (short) / 77 ms
(long). Against the incumbent Qwen worker's final baseline (61.2 tok/s,
captured just before the swap — see `docs/qwen3.6-35b-a3b-nvfp4.md`), this
is +23% decode and ~7× faster short-turn latency through the proxy hop —
an honest deployed-topology comparison, not a same-silicon A/B. Lightning's
own self-hosted MTP/DSpark and strict-tools arming remain UNEVALUATED. See
`docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md`.

**The Thor reaches it by proxy**, mirroring the pre-d1 direction reversed:
`WORKER_PEER_ORIGIN` + `WORKER_PEER_PROXY` on the Thor forward `model=worker`
to the Spark and relay the answer with `X-Lobes-Proxied-By` — validated live
the same day (`docs/evidence/2026-08-20-accept-worker-hand-spark.txt`'s
proxy-chain probe, run from the Thor's own gateway).

> **Superseded history (pre-d1, kept for the record — 2026-07-31):** the
> `worker` role was previously `unsloth/Qwen3.6-35B-A3B-NVFP4` (MULTIMODAL,
> image+video via its own ViT, self-hosted MTP draft), hosted LOCALLY on
> the Jetson AGX Thor via the `thor-worker` shape at `gpu_mem_util=0.45` /
> `max_model_len=262144`, measuring ~50.8 tok/s single-stream decode and
> 89.1% MTP acceptance
> (`docs/evidence/2026-07-31-accept-worker-thor.txt`), with the Spark
> reaching it by proxy in the OPPOSITE direction from today
> (`docs/evidence/2026-07-31-accept-worker-proxy-spark.txt`). Deviation d1
> superseded all of it in one day: different checkpoint, different
> modality, different host box, opposite proxy direction. See
> `docs/qwen3.6-35b-a3b-nvfp4.md` for that checkpoint's full history and its
> own GDN-MTP kernel gap on the fleet's newer nightly.

**`associate` — the TENTH role (opt-in hosting), the Jetson AGX Orin's local
generate lobe.** Checkpoint: `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`
— the SAME Lightning checkpoint the Spark serves as `worker`. The metaphor is
"they do, but not act": associate's responsibilities are worker's MINUS
`repo_action` — execution, ground_work, bulk_transform, drafting,
repo_inspection, run_authorized_commands, `tool_use` (forbidden:
`final_decision`, `security_decision`, `code_authoring`, `repo_action`). It
exists as a SEPARATE public address rather than a responsibilities token on
`worker` for one operator reason: the mesh may switch `worker` and `cortex`
later, and that seat has to stay free. Capability order: `hand` < `multimodal`
< `worker` < `muse` < **`associate`** < `main`/`cortex` — the highest non-cortex
rung. Third opt-in core role (`OPT_IN_CORE_ROLES`), so machine-as-brain never
hosts it and an unwired associate 404s `role_infeasible`; under pressure it
sheds 429 like cortex/senses/worker/muse, and `hand` remains the only servable
floor.

**Why sm_87 can serve an NVFP4 checkpoint at all:** Lightning's own
`hf_quant_config.json` is `W4A16_NVFP4` (WEIGHT-only, 16-bit activations) on the
experts plus FP8 on `in_proj`/`out_proj` — NOT the W4A4 activation quantization
that rules out the cortex and muse NVFP4 exports on Ampere. vLLM accepts
`quantization=modelopt_mixed` there and selects a full Marlin fallback stack
(FP8, NVFP4 GEMM, NVFP4 MoE) with native FlashAttention 2. The Orin also CLEARS
the "Warming up Mamba2 SSD Triton kernels" step that wedged the Jetson AGX Thor
indefinitely on two engine versions — that no-go is sm_110-specific.

**MEASURED live, 2026-08-26** (`docs/evidence/2026-08-26-accept-orin-associate.txt`,
the `orin-associate` shape: associate + hand + embedder + reranker, no cortex, no
senses, no audio): `ASSOCIATE_GPU_MEM_UTIL=0.56` at `max_model_len=128000` —
weights 17.81 GiB, KV 9.35 GiB, pool 1,524,000 tokens, 11.91x concurrency. Both
higher hypotheses are recorded as REFUSED rather than dropped: the vendor's 0.70,
and 0.63 (which fit the gears but not `hand`, whose 5.84 GiB resident is 59% more
than its declared util implies). Probes: known-answer PASS, multi-step reasoning
PASS, structured tool calls PASS, and unauthenticated requests — including over
the tailnet address — refused 401. Throughput, at the same depths as this board's
own llama.cpp GGUF cortex and WITHOUT speculation: **52.5 tok/s decode at 32768
depth with NO decay across 0->32768, vs 2.43 tok/s (~21.6x); TTFT 16.9 s vs
610.0 s (~36x); prefill ~1,612 vs ~64 tok/s.** NVIDIA's "89 tok/s on Jetson AGX
Orin" is deliberately NOT used as a comparator: it is an agentic-workload
aggregate with speculation, and three separate defects were found in that page's
published recipes (a nonexistent llama.cpp quantization tag, a GGUF that will not
load in NVIDIA's own Jetson-Orin image, and a vLLM DSpark repo id missing its
`-NVFP4` infix). **Still open:** DSpark is wired but default-off and UNMEASURED
on the full shape; the shape leaves only ~1 GiB free on a ZERO-swap board (see
issue #216); and no peer has addressed this lane cross-box. See
`docs/nemotron-3.5-lightning-30b-a3b-nvfp4.md`.

**`hand` — the ninth role, the fleet's FINE-TUNING BASE (default-hosted
everywhere).** Checkpoint: `LiquidAI/LFM2.5-1.2B-Instruct` (a ~1.2B hybrid
short-conv + GQA `Lfm2ForCausalLM`; 32768 native; bf16 — the catalog's
`quantization="none"` sentinel, so the lane omits `--quantization` ENTIRELY;
**text-only** — no ViT, hence no `--language-model-only`, and no thinking mode,
hence no `--reasoning-parser`; LiquidAI ships `LFM2.5-1.2B-Thinking` and a
vision variant as SEPARATE checkpoints). Requires vLLM >= 0.23.0, so the lane
pins the nightly digest, never the NGC 26.04 tag `vllm-minor` rides — verified
against the pinned digest on a physical Thor 2026-08-10 (`VLLM_VERSION
0.23.1rc1.dev672+g93d8f834d`, `LFM2_REGISTERED True`).

The metaphor is **muscle memory**: one cheap base, many LoRA adapters, each
mastering a domain. `worker` is an untrained generalist doer; `hand` is a
trained specialist. Responsibilities: `domain_mastery`, `learned_skill`,
`specialized_task`, `tool_use` (forbidden: `final_decision`, `repo_action`,
`security_decision` — v1 withholds `repo_action` deliberately, since ADDING a
responsibility later is contract-compatible while REMOVING one is a break;
granting it once adapters exist is issue #180).

Three things follow from it being cheap (~2.4 GiB bf16): it is **hosted by
every built-in shape** including the mesh-lobe ones, it is **never proxied**
(deliberately absent from all three peer channels — `NEVER_PROXIED_BACKENDS`
names that absence so a symmetry-minded refactor must delete a constant to
break it), and it is the **pressure-policy servable floor**. It also
**replaced `Qwen/Qwen3.5-4B` as the `minor`/`cheap` tier**; the 4B stays in the
catalog as a plain candidate (cite-don't-delete), still selectable via `lobes
switch`, but no tier resolves to it.

**Tool calling** uses LFM2's own `<|tool_call_start|>`/`<|tool_call_end|>`
delimiters, which are **special tokens** — the same trap that made `pythonic`
silently wrong for Gemma 4. vLLM ships a purpose-built **`lfm2`** parser whose
`__init__` RESOLVES both delimiters and **raises** when either is missing, so a
bad tokenizer revision fails loudly at startup instead of relaying a well-formed
call as prose.

**LoRA serving** ships ARMED (`--enable-lora`) with the inventory **EMPTY** —
v1 has zero adapters. Adapters are declared once in `HAND_LORA_MODULES`
(`name=path`, comma-separated), read by BOTH the engine's `--lora-modules` and
the gateway's `hand:<domain>` alias derivation so the two cannot disagree, and
fixed at boot — there is no runtime hot-load. `model=hand` serves the base and
never 404s on an empty inventory; `model=hand:<domain>` serves that adapter; an
UNdeclared `hand:<domain>` is refused with `model_not_found`, never silently
downgraded to the base. An adapter vLLM did not actually load is absent from
both `/v1/models` and `/capabilities` — verified by probing the lane's OWN
`/v1/models` rather than the gateway's filesystem, since adapter paths are
mounted into `vllm-hand`, not the gateway. Adapter PRODUCTION is `unsloth-cli`,
out of tree and one-directional (nothing under `lobes/` imports it);
`agentculture/unsloth-cli#16` tracks LFM2.5 support there.

Budgets are **DECLARED, not measured** (#108) on every card. See
`docs/lfm2.5-1.2b-hand.md` and `docs/colleague-stack.md`, whose `hand` section
also records why **adding a role is effectively irreversible** — read it before
proposing a tenth.

An opt-in **realtime audio overlay** (`lobes init --fleet --audio`) adds an OpenAI
`/v1/audio/*` facade — a `realtime` bridge container (shipped in the wheel as
`lobes.realtime`) that the gateway fans `/v1/audio/*` out to — backed by two
fixed GPU sidecars: **Parakeet** STT (`nvidia/parakeet-tdt-0.6b-v2`, NeMo ASR →
`POST /v1/audio/transcriptions`) and **Chatterbox** TTS (Resemble AI, 0.5B,
Apache-2.0 → `POST /v1/audio/speech`, 24 kHz, zero-shot voice cloning; it replaced
the retired Magpie NIM — no NGC key). These two are hardcoded, **not** in the
switchable catalog (`lobes/catalog.py`).

**`GET /v1/realtime` — the server_vad WebSocket session (issue #149).**
Beyond the batch facade, one WebSocket connection replaces separate STT
calls with a persistent session: stream PCM16 mono little-endian in
(**24000 Hz default, 16000 Hz accepted** — the server resamples down to
16 kHz itself) and receive `session.created` /
`input_audio_buffer.speech_started`/`...speech_stopped` /
`conversation.item.input_audio_transcription.completed` / `error` events
back on the SAME connection (event schema + config parsing in
`lobes/realtime/_session.py`; the server_vad segmenter — a pure state
machine, Silero injected as a callable — in `lobes/realtime/_segmenter.py`;
the thin FastAPI route wiring both to real Silero + scipy in
`lobes/realtime/app.py`). A never-silent turn force-commits at
`VAD_MAX_TURN_MS` (default 30s) with `reason="max_turn"` — a normal
boundary event, never an `error` (true of the server's behaviour from the
start, but **not observable by any client** until #151 put `reason` on the
wire). Sessions are ephemeral (no resume; any
disconnect tears the session down and a reconnect gets a brand-new session
id) and reached **through the gateway** (101-upgrade + byte tunnel,
`lobes/gateway/_realtime.py`) under the same opt-in `GATEWAY_API_KEY`
bearer gate; a declared-off `stt` lane 404s `role_infeasible` and the
session is **never** proxied cross-box (the #129 proxy-lobes forwarder is
POST-only). This redeems two in-tree IOUs — `app.py`'s own "PR2 adds the
`/v1/realtime` WebSocket route" docstring promise, and
`realtime-pipeline.md`'s former "planned for a later release" boundary
claim — against the #149 baseline probe (the deployed facade served four
batch routes and no WebSocket, forcing reachy-mini-cli's client-side
energy-threshold endpointing). The **session mechanism** is VALIDATED live
on the DGX Spark GB10,
2026-07-21 (`docs/evidence/2026-07-21-accept-realtime-spark.txt`): a full
session through the gateway tunnel against real Silero + Parakeet, at both
24000 Hz and the 16000 Hz passthrough. The live run is also what caught the
tunnel's leftover-direction bug — the bridge's first frame was being sent
back upstream, killing every session the instant it opened, and the unit
test had asserted that wrong direction as correct. That transcript predates
the #151 wire change below, so it proves the tunnel, the VAD and the STT
forward — not the wire the server speaks today. Still UNVALIDATED, and NOT
retired by #151: a real
microphone (the runs used synthesized audio), the VAD-unavailable path,
concurrent sessions, and the max-turn cap.

**Voice-to-voice on the same socket (issue #151) — a coordinated wire break
plus an opt-in conversation surface.** Two changes land together, and the
first is a **break**: raw binary audio frames are gone. Audio now travels as
OpenAI-shaped **base64 JSON events in BOTH directions** —
`input_audio_buffer.append` inbound, `response.audio.delta` outbound (codec:
`lobes/realtime/_wire.py`). A binary frame now yields the new named
`invalid_wire_event` error instead of being read as audio. The deployed
reachy-mini-cli speaks the old wire and **cannot stream until it adapts**
(tracked in **reachy-mini-cli#115**); that is a recorded, operator-accepted
decision (frame decision c40), not a regression. Second, **conversation is
opt-in**: a session answers nothing until the client sends `response.create`,
and one that never does emits exactly the #149 transcription-only sequence
(the ears-only contract reachy depends on — a structural property of
`_conversation.py`, where every floor call sits behind `if self.armed`).
Once armed, a committed turn is generated through the gateway's own
`/v1/chat/completions` (voice lane defaults to `multimodal`, ~1 s to a short
reply; `OPENAI_MODEL` overrides; a lane this box does not host surfaces as
`generate_failed` with `hosted_by`, never a silent fallback), synthesized by
Chatterbox and streamed back as 24 kHz deltas that never resample. Speaking
during playback is *meant to* **interrupt**: generate and TTS are both
cancelled, the undelivered remainder is never sent, and
`response.interrupted` goes out — only the plausibly-heard prefix enters
history. That contract is implemented in `_floor.py` and offline-proven,
but the 2026-07-22 live run found it **inert**: the route delivered audio
as fast as the socket drained (measured 2–4 ms for 7.5–8.5 s of speech),
so the floor left `SPEAKING` before the user had heard two words and a
barge-in landed while it was already `LISTENING` — a new turn opened and
`response.interrupted` never fired. 0.54.1 paces delivery to the playhead
(`delivery_pause_ms`, ≤ 400 ms lead) to close that gap; barge-in itself
is still UNVALIDATED live (#108). New machinery:
`_floor.py` (explicit floor/turn state machine, per-stage 60 s deadlines →
`response_timeout` rather than a wedged session), `_turn.py` (generate
payload shaping), per-session in-memory history + system prompt, and
per-lane TTS pools so a spoken reply never queues behind batch TTS. New
knobs: `BARGE_IN_WINDOW_MS` (750; a guard window, not a delay),
`BARGE_IN_MODEL` (threaded but **unconsumed** — window-only barge-in ships),
`TTS_VOICE_CONCURRENCY` (1), `DEFAULT_SYSTEM_PROMPT`. Boundary events now
carry `at_ms` and `reason`, so VAD tuning is observable live. A **local-only**
Astro harness under `site/` drives all of it from a browser (real mic, live
event stream, audio out) through a local credential-injecting WebSocket proxy
reached via `ssh -L` — never deployed; CI only builds it. **PARTIALLY
validated (#108):** every decision lives in stdlib modules the offline
suite covers and `app.py` stays a `pragma: no cover` shell. The live
acceptance transcript has now landed
(`docs/evidence/2026-07-22-accept-realtime-voice-to-voice-spark.txt`,
2026-07-22, DGX Spark GB10) and it is **split** — voice-to-voice works
end-to-end (armed session → generate → Chatterbox reply streamed back as
deltas), and **barge-in does not**, for the delivery-pacing reason above.
Treat that transcript as evidence for the conversation loop only; it
predates the 0.54.1 pacing fix, so it is not evidence for barge-in either
way. Still UNVALIDATED: barge-in (re-run needed against 0.54.1), a real
microphone (the run used synthesized audio), concurrent sessions, and the
VAD-unavailable path. On muting, note the deliberately narrow rule
(deviation `d1`): **automatic** mute-during-playback stays FORBIDDEN — it is
the AEC substitute that makes barge-in impossible, since you cannot interrupt
a machine that has stopped listening — while **user-initiated** mute/mic-off
is allowed, because AEC is genuinely owned at the client edge (Reachy
firmware, browser `echoCancellation`).

See `docs/realtime-pipeline.md`, `docs/parakeet-stt.md`,
`docs/chatterbox-tts.md`, `docs/gateway-fleet.md` (the realtime lane), and
`docs/openai-api.md` (the full
OpenAI-compatible endpoint surface, including `/v1/realtime`). `lobes
explain realtime` / `api` are the in-CLI versions.

## Machine profiles and supported hardware

lobes runs the fleet with knob values tuned to the hardware it lands on.
**Machine profiles** — built-in TOML declarations in `lobes/profiles/builtin/` —
declare per-role models, context lengths, GPU memory budgets, attention
backends, and vLLM knobs. `lobes init` auto-detects the card via `nvidia-smi` +
hostname, resolves a profile by name, and renders it to env vars the compose
template substitutes at startup.

**Validated support:**

| card | profile | status | validation |
|---|---|---|---|
| **DGX Spark** (Grace Blackwell, 128 GB unified) | `spark` | load-tested | 2026-06-03 — fleet duo (cortex 128K + senses 32K) serves at ~7.8–8.0 tok/s decode (27B primary, util 0.30) with FlashInfer attention. The correctness probes postdate that run and are unverified on the GB10 (rerank ordering: issue #106). See `docs/tuning-profiles.md`. |
| **Jetson AGX Thor** (Blackwell-class sm_110, 128 GB unified) | `thor` | load-tested | 2026-07-13 — the three correctness probes pass (cortex known-answer, embed ranking, rerank ordering) with four validated divergences: `cortex kv_cache_dtype=auto` (uncalibrated-fp8 exposure, #109), `embedder`/`reranker attention_backend=TRITON_ATTN` (FLASH_ATTN pooling broken on sm_110, #105), `reranker enforce_eager=true` (CUDA graphs unstable on sm_110). Concurrent first boot can fail on a memory race — see the boot-ordering caveat in `docs/machine-profiles.md`. |
| unknown card | `base` | conservative fallback | — small 4B model, no 27B, no multimodal (senses disabled) to avoid OOM on first boot. Resolved when card detection returns UNKNOWN. See issue #107 (broader tuned-small-model work, future). |

**Custom profiles:** operator-defined TOML files in `<deploy-dir>/profiles/<name>.toml`
override built-ins by name. See `docs/machine-profiles.md#writing-your-own-profile` for
the format, and `lobes explain profiles` for the brief reference.

**See also:** `docs/machine-profiles.md` (the deep reference: detection flow, knob
meanings, Thor's validated divergences, custom profiles, goldens contract);
`lobes explain profiles` / `lobes explain tuning` (in-CLI).

## Deployment shapes

Orthogonal to the machine-profile axis above (how a role is *tuned* on a
card) is the **deployment-shape** axis (issue #113): which of the eight
Colleague roles a box *hosts* at all, composed as pure data over the card
profile at render time (`lobes/profiles/shapes.py`, `shape_render.py`).
**machine-as-brain** (the default — bare `lobes init`, unchanged, zero new
decisions) hosts every role a card can serve; the four core roles stay
default-on **by machine-as-brain**, not unconditionally — a mesh-brain shape
drops one heavy generate lobe to a peer box via a *generated*
`docker-compose.shape.yml` override (the base fleet template itself stays
unconditional). Two mesh-lobe shapes are validated live (2026-07-14):
**`spark-lobe`** (DGX Spark GB10 — drops `senses`, `cortex` reclaims to
`gpu_mem_util=0.44` / `max_model_len=262144`, measured KV pool 888,946
tokens / 3.39× concurrency at full 256K) and **`thor-lobe`** (Jetson AGX
Thor — drops `cortex`, `senses` reclaims to `gpu_mem_util=0.30` /
`max_model_len=131072`, measured KV pool 1,418,554 tokens / 10.82×
concurrency at 131072). Both reclaim values are *measured*, not computed —
the naive reclaim-sum/solo-default was refused by vLLM on the live,
unified-memory box in each case. A fourth built-in shape, **`orin-small`**
(mesh-brain end-state, issue #112, t2), drops BOTH heavy lobes and hosts the
opt-in `minor` gear (`vllm-minor`) instead, alongside the pooling gears and
audio overlay — it ships as **declared, UNVALIDATED data only** (the #108
rule: no physical Jetson AGX Orin has booted it, so no doc, support table,
or `lobes capabilities` output may claim it validated). A fifth built-in
shape, **`thor-muse`**, likewise drops BOTH heavy default lobes and instead
hosts the opt-in `muse` lobe (`vllm-muse`, Gemma 4 31B) plus the pooling
gears and audio overlay — `muse` is an **opt-in core role**
(`OPT_IN_CORE_ROLES`): hostable only by an explicit shape, with the full
muse declaration (model + budget knobs) in the shape's own overrides, the
card profiles silent on it, and `base.toml` vetoing it; `thor-muse` too is
**declared, UNVALIDATED** — its budget values (`gpu_mem_util=0.55`,
`max_model_len=262144` — the full 256K window) were measured by the
2026-07-17 live boot on a physical Thor (the 0.40 hypothesis was refused),
and it stays UNVALIDATED until the acceptance transcript lands under
`docs/evidence/` (#108). **`thor-muse` is now DORMANT/unhosted** — the
physical Thor moved to hosting `worker` instead (below) and no box declares
`MUSE_PEER_ORIGIN` — but the shape file, its TOML, and its goldens stay
in-tree unchanged (cite-don't-delete); nothing here or in `lobes
capabilities` claims a box currently renders it. `worker` — the eighth
Colleague role — introduced a **second opt-in core role** on the same
mechanism: the **`thor-worker`** shape drops BOTH `cortex` and `senses` and
instead hosts the opt-in `worker` lobe (`vllm-worker`, Qwen3.6-35B-A3B) plus
the pooling gears and audio overlay, mirroring `thor-muse`'s structure exactly
(`OPT_IN_CORE_ROLES = ("muse", "worker")`, `base.toml` veto, shape-owned
override declaration). It has **LANDED and is VALIDATED** on the physical
Jetson AGX Thor, 2026-07-31, with **measured** values —
`gpu_mem_util=0.45` at the full `max_model_len=262144`, KV pool 41.78 GiB =
14.07x concurrency, MTP acceptance 89.1%
(`docs/evidence/2026-07-31-accept-worker-thor.txt`); the 0.45 hypothesis booted
first try, unlike `thor-muse`'s refused 0.40. Select with `lobes init --shape
<machine-as-brain|spark-lobe|thor-lobe|orin-small|thor-muse|thor-worker>`
(dry-run by default, `--apply` to commit, byte-for-byte restorable by re-running
with the previous shape). A dropped role is flagged `feasible:false` on both
`lobes capabilities` and `GET /capabilities`, omitted from `/v1/models`, and
404s `role_infeasible` on every alias — never half-served. Opt-in **honest
referral** (issue #112, t3): declaring a peer origin per dropped role
(`PRIMARY_PEER_ORIGIN` / `MULTIMODAL_PEER_ORIGIN` / `MUSE_PEER_ORIGIN` /
`WORKER_PEER_ORIGIN` / `EMBED_PEER_ORIGIN` / `RERANK_PEER_ORIGIN` — plus,
since #129, the first-class audio lanes `STT_PEER_ORIGIN` / `TTS_PEER_ORIGIN`,
declared off per-lane with `STT_FEASIBLE`/`TTS_FEASIBLE=false` — always
operator-typed, never derived, per #92) makes
both capabilities surfaces and the `role_infeasible` 404 body name the
hosting peer (`hosted_by`); by default this is annotation only — the gateway
does not forward a request to a peer on the origin declaration alone, and
with no peer config every response is byte-identical to the pre-referral
contract. A box can opt into actually following its own referral — see
proxy-lobes, next.

**The mesh-brain end-state (issue #112)** — one heavy lobe per box, cheap
gears co-reside, the brain stays whole across the mesh — has landed on top of
the near-term work above, recording four decisions: (1) cross-box
reachability is **direct addressing + opt-in honest referral** by default,
now with an opt-in proxy extension (below); (2) the cheap
gears (`embedder`/`reranker`/`stt`/`tts`) **co-reside** on every box that
wants them — no gear is forced to move; (3) the reference shape assignment is
**Spark GB10 = `cortex` via `spark-lobe`, Thor 128GB = `senses` via
`thor-lobe`, Orin 64GB = small-model lobes via `orin-small`**; and (4) the
shape axis is **mixable** — specialized, multi-role, and mixed boxes (local
or cloud) compose into one brain, with `machine-as-brain` staying the default
and one-box users unaffected. The referral surface is live-validated
cross-box on the physical Thor
(`docs/evidence/2026-07-14-accept-referral-thor.txt`); physical Orin
validation remains open. See `docs/deployment-shapes.md` (the deep
reference) and `lobes explain shapes` (in-CLI).

**Proxy-lobes (issues #115/#127, phase 1 — landed on top of referral).** A
dropped role can go beyond referral-only to a third state, **proxy**: this
box forwards the request to its declared peer instead of 404ing, so the
caller never has to know it moved. Two knobs, both opt-in and both required
together: `<PREFIX>_PEER_PROXY=true` (arms the forward; inert without a
declared `<PREFIX>_PEER_ORIGIN`) and `<PREFIX>_PEER_API_KEY` (the outbound
credential — always **a copy of the peer's own inbound `GATEWAY_API_KEY`**,
never a value minted per pairing, so key material scales **O(machines)**,
not O(pairs)). The caller's own `Authorization` (validated by this box's own
opt-in `GATEWAY_API_KEY` inbound gate) is stripped before every forward and
never reaches a peer. Proxying is single-hop — a request that arrives already
marked `X-Lobes-Proxied` is refused (`508 proxy_loop`) rather than re-forwarded
— and every proxied answer carries `X-Lobes-Proxied-By: <peer origin>` so a
caller can always tell a forwarded answer from a locally-served one. Peer
origins are assumed reachable over a private/tailnet transport, never the
public internet (no TLS termination happens at this layer). With no
`<PREFIX>_PEER_PROXY` set anywhere — every pre-#115 deployment — every response
stays byte-identical to the pre-proxy contract.

**Live as of 2026-07-31:** the DGX Spark (`spark-lobe`) proxies TWO roles —
`senses` → the AGX Orin and `worker` → the Jetson AGX Thor — so a caller
addresses either on the Spark's own gateway and never dials the peer box.
Both answer 200 with `X-Lobes-Proxied-By`, image input included. **A proxied
role's `ready` and `context` are the PEER's own advert (issue #220):** the
background probe reads the peer's `GET /capabilities` and relays that role
entry, falling back to the old `/v1/models` served-id check only for a peer
with no such entry. Both halves fixed a measured lie — on the Spark,
2026-08-27, `associate` advertised `ready:false, context:1048576` against an
Orin reporting `ready:true, context:128000` — with two distinct causes: the
`/v1/models` check is unsatisfiable for a box forwarding an ALIAS (which
`associate` must do, sharing a checkpoint with `worker`), and a role this box
does not host has no `<PREFIX>_MAX_MODEL_LEN` to read, so context fell through
to the catalog's native ceiling. That collision is also why **role-name
addressing (`model=associate`) is the documented proxied path** — the raw
checkpoint id is ambiguous and resolves to the local `worker` lane. Note a
proxied role reports `feasible: false` **by design** (it means "this box does not *host*
it", not "you cannot use it here"); `proxied: true` + `hosted_by` + `ready` are
the fields that say it is usable, and `loaded` is a *wiring* fact, not a
running one. See `docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in`
and `docs/deployment-shapes.md#following-the-referral-proxy-lobes-opt-in`.

**The cortex replica pool (issue #199) — landed and VALIDATED live
2026-08-25 on the Spark+Thor NVFP4 pair, cortex-only.** Where proxy-lobes forwards a *dropped* role to its one peer,
a replica pool lets a box that already HOSTS a role forward *some* of that
role's requests to an equally-compatible peer replica when the peer is less
loaded — the mirror case, and a property of the awake/proxy states, not a
fourth one. Declaring the plural `<PREFIX>_PEER_ORIGINS` (positionally
paired with `<PREFIX>_PEER_API_KEYS`, an empty key slot legal) beside the
existing singular `<PREFIX>_PEER_ORIGIN`, plus an operator-typed
`GATEWAY_SELF_ORIGIN`, arms a background `ReplicaCache` that live-probes
each peer's `GET /status` (load) and `GET /capabilities` (fingerprint) and
picks the least-loaded compatible, ready, non-busy replica — local wins
ties, an `X-Lobes-Affinity` header stickies within a margin, and every
pooled answer carries `X-Lobes-Served-By` (local) or `X-Lobes-Proxied-By`
(forwarded) plus `X-Lobes-Route-Reason` on both. Under local pressure a
pooled request forwards to a selectable peer instead of shedding 429; only
"no replica anywhere is selectable" still sheds, and at most one forward
happens per request. Compatibility is a live-probed fingerprint (served id +
quantization + max context + runtime), never the catalog — `kv_cache_dtype`/
parsers/speculative config are informational only. With no
`*_PEER_ORIGINS` declared, every response stays byte-identical to the
pre-pool contract. **Status: VALIDATED live (#108)** — `docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`:
three concurrent requests to the Spark front, with the Spark under organic
iowait pressure, were all served by the Thor at **19.1 tok/s aggregate vs
the 11.0 tok/s single-owner baseline (+74%)** with
`X-Lobes-Route-Reason: local-busy-forwarded`; with the Thor's gateway
stopped the Spark kept serving alias and raw id (`sole-ready`); a marked
arrival at the Thor never re-forwarded. Two divergences are recorded, not
hidden: a **raw-id request under local PRESSURE is not forwarded** (the
pressure gate is tier-alias-only, #85 → issue #215; under load alone raw id
and alias place identically), and affinity yields whenever the preferred
replica's box flips busy. The drafter difference (DSpark vs MTP) is NOT
visible in the fingerprint yet because neither box declares
`PRIMARY_SPECULATIVE_CONFIG` in `.env` (#214). The pre-pool baseline is captured
(`docs/evidence/2026-08-25-baseline-cortex-single-owner.txt`: an 8-way
flood of raw-id requests to one gateway queued at 11.0 tok/s aggregate —
the same as a single request — while the peer idled at `running=0`, and
organic iowait pressure shed three concurrent `model=cortex` alias
requests 429 without consulting it), Validated scope is `cortex` on the Spark+Thor
NVFP4 pair only — the Orin's llama.cpp cortex is exempt (a separate
candidate), and any other pooled role (senses/muse/worker/embedder/
reranker/hand/stt/tts) is declared/unvalidated data only, even though the
plural peer family is generic across all nine role prefixes. See
`docs/gateway-fleet.md#replica-pools-one-lobe-n-replicas-opt-in-cortex-validated-only`,
`docs/deployment-shapes.md`, and `docs/colleague-stack.md` (capabilities
schema: additive `replicas`/`fingerprint` fields).

**Peer-only pools — a role the box hosts NOWHERE (DECLARED, not validated).**
The pool above forwards a role a box *hosts*; a box that hosts it nowhere took
the referral/proxy branch, which dials the **singular**
`<PREFIX>_PEER_ORIGIN` and therefore pins every request to one peer. Measured
on the Jetson AGX Orin 2026-08-30: every `model=cortex` request answered 200
with `X-Lobes-Proxied-By` naming the same peer, while the Spark and Thor each
published two compatible ready cortex replicas. Declaring the plural family on
a `FEASIBLE=false` role now places each request across the declared replicas
instead. Four rules, each a recorded decision: **peers agree with each other**
(no local lane means no reference, so the first READY peer in DECLARATION
order supplies the fingerprint every other peer is compared to, published as
`reason: "fingerprint reference"`; disagreement leaves nothing compatible —
#199 h11 restated); **the singular origin is REQUIRED** (`hosted_by` reads it,
so plural-without-singular is refused at startup with a named
`ReplicaConfigError` — a pool on a role the box HOSTS needs no singular
origin, publishing no referral at all); **never worse than today** (nothing
selectable falls through to the existing singular forward, and with no
singular origin the 404 `role_infeasible` is byte-identical); and **the
singular credential is inherited** by the replica whose origin IS the singular
peer, since the two key channels parse independently and an upgraded box would
otherwise start sending no `Authorization` to a peer it was already
authenticated to. `/capabilities` folds the advert across the set (`ready` =
any compatible replica ready; `context` = the fingerprint-agreed window, not
the catalog ceiling) and `/v1/models` lists a pooled dropped role on that same
evidence; `feasible` stays `false` — pooling never makes a box a host. Local
pressure is not re-applied and the single-hop 508 guard is unchanged.
**Status: DECLARED (#108)** — unit-proven in
`tests/test_gateway_peer_only_pool.py`, spec/plan under
`docs/specs/2026-08-30-peer-only-replica-pools.md` and
`docs/plans/2026-08-30-peer-only-replica-pools.md`, with **no** live
cross-box acceptance transcript yet.

## The deployment lock and the variation catalog

Orthogonal again to profile (how a role is tuned) and shape (which roles a
box hosts) is **what a box actually runs**: the compose files, overrides and
Dockerfiles on disk, which drift from the packaged templates the moment
anyone hand-edits them. The **deployment lock** captures that as a committed
artifact. The motivating incident is on the record, not remembered: on
**2026-08-25**, preparing the cortex replica pool (#199), the Spark's
`docker-compose.yml` was found hand-edited with the DSpark
`--speculative-config` baked into `vllm-primary` while the Thor's equalled
the template — only a live diff revealed it, the Spark could not be safely
re-scaffolded (the pool's passthrough lines went into
`docker-compose.override.yml` instead), and the two boxes' live fingerprints
both read `speculative_config: unknown` while genuinely drafting differently
(`docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`, "Deploy
record"; issue #214).

- **`deployment.lock.toml`** (`lobes/runtime/_lock.py`) — `[variation]`
  (machine type or setup, NEVER a hostname — `lobes/variation.py`), `[env]`
  and `[files]` (name → `sha256:` digest). `[env]` is an **allowlist derived
  from `lobes/profiles/render.py`'s own tables**, never a denylist-redacted
  copy of a deployed `.env`, so a credential cannot enter by someone
  forgetting to blank a line — narrowed by two exclusions (**deviation d1**):
  `COMPOSE_PROFILES` (operator-typed) and any `_URL`-suffixed key (wiring).
- **`lobes init --from-lock <dir|file>`** — a distinct SOURCE, not a fourth
  renderer input: it bypasses profile/shape resolution and materialises the
  committed files verbatim (refused alongside
  `--single`/`--audio`/`--profile`/`--shape`). `.env` stays **merge-only** —
  existing lines are never rewritten, only missing locked knobs appended. A
  machine-type mismatch **refuses** (an UNKNOWN card counts as one) because
  the bypass also skips the card's csv-vs-devices GPU-access correction; the
  override is its own flag, `--allow-variation-mismatch`, never `--force`.
- **`lobes doctor` → `lock_drift`** — names the SPECIFIC differing files and
  locked keys; no lock present means no finding at all; read-only, so
  `--fix`'s never-rewrite-an-existing-`.env`-line convention is untouched.
  `lobes switch` warns whenever a lock is present. Note **deviation d4**: the
  spec's claim that `switch` writes the locked key family is WRONG (it writes
  only legacy `VLLM_*` keys, which never intersect the allowlist) — staleness
  is caught by tracking `.env`'s own **digest** in `[files]` instead.
- **`deployments/`** — the variation catalog, published for machine types
  this operator may not run. Each `<id>[__<shape>]/` carries its lock, its
  files and a `VARIATION.md` whose `## Measured result` either cites an
  existing `docs/evidence/` transcript or states verbatim `No measured
  result.` — never both, never neither (`lobes/variation_catalog.py`).
- **Secrets** — a positional `.gitignore` rule (any name *ending* `.env` is
  ignored; a `.env.` *prefix* is tracked, with a `!tests/goldens/**/*.env`
  negation), a `.secrets.env` sibling read by every fleet service as a second
  `required: false` `env_file` entry, and a **required** CI job
  (`secrets-scan` → `scripts/scan_deployment_secrets.py`) over the lock and
  every verbatim-committed compose/override/Dockerfile — the half the
  allowlist cannot protect. Leak recovery: `docs/secret-rotation.md`.

**Honesty (#108) — read this before citing any of it as working.** **No real
box has been captured**: `deployments/` ships a README and a template and
ZERO variations, and every catalog behaviour is exercised against fixtures
under `tests/fixtures/deployments/`. **There is no capture verb** — the lock
writer is a library with no CLI caller, so every "re-capture the lock"
instruction means calling it. **Serve-after-restore is unmeasured**: the
mechanism guarantees byte-identical files and a merge-only `.env` (both
test-proven), but no box has been restored and then served. A lock-restored
**fresh** box is not yet servable (**deviation d2**): the fleet compose
bind-mounts `mg-logwrap.sh` and the tool-parser plugin, which are packaged
SCAFFOLD files, so a lock naming only compose + Dockerfiles restores an
incomplete deployment. The buildability preflight (`_buildability.py`, run by
`--from-lock`) is **offline and warn-only** and cannot prove a wheel
uninstallable — its raising path needs an index query nothing wires
(**deviation d3**). All four deviations are `proposed`, not approved. See
`docs/deployment-lock.md` (the deep reference) and `lobes explain lock`.

## Deployment model

lobes is **scaffold-based, not checkout-based.** The canonical
`docker-compose.yml` + `env.example` are packaged under `lobes/templates/`
and shipped in the wheel. `lobes init` materialises them into a deployment dir —
default **`~/.lobes`**, or a `TARGET` path, or `.` for the local folder.
Every model-ops verb resolves the deployment dir as: `--compose-dir` →
`$LOBES_DIR` → `~/.lobes`, falling back to the legacy `$MODEL_GEAR_DIR` →
`~/.model-gear` when those are set / already scaffolded (so a pre-rename
deployment keeps working). There is no compose file at the repo root.

## CLI surface

```text
lobes/                 # Python package (pip install lobes-cli)
├── __init__.py             # __version__ via importlib.metadata("lobes-cli")
├── __main__.py             # python -m lobes
├── assess.py               # correctness probes + throughput/prefill (stdlib urllib)
├── catalog.py              # the supported-model catalog (the switchable "gears")
├── templates/              # packaged docker-compose*.yml + env.example + Dockerfiles (lobes init)
├── runtime/                # _env (.env r/w) · _compose (dir resolve + docker) · _health · _tunnel (cloudflared)
├── gateway/                # stdlib OpenAI-compatible reverse proxy (the fleet front)
├── realtime/               # /v1/audio/* facade: bridge · tts_client · chatterbox_server · _readiness
├── explain/                # markdown catalog for `lobes explain <path>`
└── cli/
    ├── __init__.py         # argparse main(); registers every verb
    ├── _errors.py          # ModelGearError + EXIT_USER_ERROR / EXIT_ENV_ERROR
    ├── _output.py          # strict stdout/stderr split; --json result emitter
    ├── _runtime_ops.py     # shared glue (deployment dir, port, compose_check)
    └── _commands/          # one module per verb: register(sub) + handler
        ├── switch.py serve.py stop.py status.py assess.py benchmark.py init.py fleet.py
        └── logs.py tunnel.py whoami.py learn.py explain.py overview.py doctor.py cli.py
```

**Lifecycle (turn on / off):** `lobes serve` (alias `start`) brings the default
deployment **up** (`docker compose up -d`, then waits for `/health`) — since #69
`lobes init`/`serve` default to the **main + multimodal duo** (the legacy
single-model scaffold is opt-in via `lobes init --single`/`--legacy`). `lobes
stop` takes it **down** (`docker compose down` — it *removes* the containers, not
a pause). The fleet lane mirrors this: `lobes fleet up` (`up -d --build`) / `lobes
fleet down`. `lobes switch <model>` is a down+up with a model swap. `lobes status`
/ `lobes fleet status` observe without mutating.

**Mutation safety:** write verbs (`switch`, `serve`, `stop`, `init`, `fleet up`,
`fleet down`, `tunnel`) default to **dry-run**; require `--apply` to commit. Agents
call CLIs in loops, so safe-by-default is mandatory. The read-only verbs (`status`,
`assess`, `benchmark`, `logs`, `overview`, `whoami`, `explain`, `doctor`) never
change the world — with one opt-in exception: `doctor --fix` is doctor's write
lane (#119), and it follows the same convention (`--fix` alone prints the
missing-only heal plan; `--fix --apply` writes absent files/keys, never
rewriting an existing `.env` line).

## Build / test / publish

- **Install for dev:** `uv sync`
- **Run CLI from source:** `uv run lobes --version` / `uv run python -m lobes whoami`
- **Tests (all):** `uv run pytest -n auto -v`
- **Single test:** `uv run pytest tests/test_cli_runtime.py::test_name -v`
- **Lint:** `uv run black --check lobes tests`, `uv run isort --check-only lobes tests`, `uv run flake8 lobes tests`, `uv run bandit -c pyproject.toml -r lobes`
- **Rubric gate:** `uv run afi cli doctor . --strict` (CI blocks merge if it fails).
- **Version bump (required every PR):** `python3 .claude/skills/version-bump/scripts/bump.py {patch|minor|major}` — updates `pyproject.toml` and prepends a CHANGELOG entry. The `version-check` CI job **fails the PR if the version equals main's** (AgentCulture every-PR-bumps rule — no exceptions, even for docs/config-only changes). Version is the single source of truth in `pyproject.toml`; `lobes.__version__` is read from package metadata at import.
- **Publish:** push to `main` → `publish.yml` builds with `uv build` and publishes `lobes-cli` to PyPI via Trusted Publishing (no API tokens); `model-gear` is published as a deprecated alias that redirects to `lobes-cli`. PRs publish a `.dev<run_number>` to TestPyPI. Fork PRs are skipped (no OIDC).

## Skills convention

Six skills are vendored from steward (the canonical upstream) under
`.claude/skills/<name>/`: **`cicd`**, **`communicate`**, **`version-bump`**,
**`run-tests`**, **`sonarclaude`**, **`doc-test-alignment`**. This is
*cite-don't-import*: copies are owned by this repo and may diverge from steward.

Three more are vendored from **`agentculture/devague`** (re-broadcast via
guildmaster) — the idea→spec→plan→implementation operator chain for the
deterministic `devague` CLI: **`think`** (idea→spec), **`spec-to-plan`**
(spec→plan), and **`assign-to-workforce`** (plan→parallel implementation). These
three carry **`type: command`** in their frontmatter — load-bearing on the
culture/agex backend (a `SKILL.md` without `type:` is silently skipped when the
repo declares an agent in `culture.yaml`). They depend on the `devague` CLI at
runtime (`uv tool install devague`), resolved portably by the wrappers.

One skill is **local to this repo** (not vendored): **`model-runner`** — a thin
pointer/shim to the `lobes` CLI for switching/serving/assessing the model. The
real implementation is the `lobes` package; the shim `exec`s `lobes`.

The provenance of every vendored skill (citation path + authoring origin) is
recorded in **`docs/skill-sources.md`**.

Each skill ships:

1. `SKILL.md` — *why* and *when* to use it (frontmatter `name` must equal the
   directory name; short prose, no inline 10-step walk-throughs).
2. `scripts/<entry-point>` — the script that automates the workflow.
3. **No external path dependencies.** Scripts must not reach outside this repo.

Per-machine paths live in **`.claude/skills.local.yaml`** (git-ignored); a
committed **`.claude/skills.local.yaml.example`** documents every key. Skills
read the local file and fall back to the example. (The Culture posting nick is
`lobes` — the deployed agent shares the repo/tool name.)

## PR workflow

Every task gets its own branch and PR. Before merging:

1. Wait for all reviewer comments (Qodo, Copilot, humans).
2. Fix valid findings — commit to the same branch.
3. Push back on invalid findings — reply with reasoning.
4. Reply to every thread (fix confirmed or pushback explained).
5. Resolve all threads. **Never merge with unaddressed review comments.**

Bump the version (above) on every PR or CI's `version-check` job fails the run.

## Working with the mesh from here

- **Culture CLI:** `culture` — server lifecycle, agent start/stop, mesh linking.
  Path references assume siblings are checked out alongside this repo
  (`../culture`, `../daria`, `../steward`).
- **steward owns the six steward-sourced skills** (the devague trio is owned
  upstream by `agentculture/devague`; see `docs/skill-sources.md`) and the
  sibling-pattern contract.
  steward files issues on siblings but never edits them — so scaffolding and
  alignment work *for this repo happens in this repo*. steward's
  `docs/skill-sources.md` "Downstream copies" column may still list this repo
  under a retired name (`lepenseur` or `model-gear`); fixing that is a **PR on
  steward**, not an edit from here.

## Conventions and workflow

**Memory discipline — recall before, remember after.** This repo keeps its
eidetic memory **in-repo and public**: records resolve to
`<repo-root>/.eidetic/memory` — committed, and shared with the team and mesh
peers (the `claude` and `colleague` backends both read the same
`lobes` scope), so memory travels with the repo, not a private
home-dir store. Make it a per-task habit:

- **`/recall` before you start.** Search the store for the area you're about
  to touch — prior decisions, gotchas, "have we done this before?" — so you
  build on what's already known instead of re-deriving it. Do this before
  non-trivial tasks, not just when asked.
- **`/remember` when something worth keeping surfaces.** A non-obvious
  decision and its rationale, a constraint, a fix and *why* it was needed, a
  gotcha that cost time, a fact the next session would otherwise re-learn.
  Capture it as it happens, not at the end when it's faded.

A plain `/remember` lands the note in `./.eidetic/memory` in this repo — no
flag needed (the wrappers here default to `--visibility public`; in-repo
routing needs `eidetic >= 0.10.0`, older CLIs keep records in `$HOME`). Keep
something out of the committed store only by passing `--visibility private`
(routes to `$HOME/.eidetic/memory`, never committed); `/recall` reads both
stores and merges. Don't store what the repo already records (code structure,
git history, what's already in this file or `CHANGELOG.md`) — store what you'd
have to re-derive. These are the `recall`/`remember` skills (`.claude/skills/`),
backed by the `eidetic` store.
