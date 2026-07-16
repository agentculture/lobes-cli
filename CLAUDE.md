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

The served model is **`vllm-local/sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`** (a
Qwen3.6 27B with hybrid Mamba/linear-attention layers, re-exported with its MTP
draft head restored so vLLM speculative decoding (Multi-Token Prediction) works;
text-only (ViT vision tower removed), NVFP4, 256K native; thinking mode with a
reasoning trace; ~2.4x single-stream decode over the archived baseline). This is
the **`cortex`** role — the fleet's reasoning/deciding/final-authority lobe
(issue #81). **Served context depends on deployment shape:** the legacy single-model
scaffold (`lobes serve`, no fleet) still serves the full 256K solo; the default
**fleet** duo serves `cortex` at **128K** (`PRIMARY_MAX_MODEL_LEN=131072`) so it
can co-reside with the multimodal gear — see "Colleague roles" below and
`docs/colleague-stack.md#migration-before--after` for the full before→after
table. lobes runs it; the `acp` `vllm-local` provider connects the lobes agent to
it. (It is the fleet's default primary/`cortex`. `mmangkad/Qwen3.6-27B-NVFP4` is
the archived former primary, demoted to a candidate but kept — it is the
tokenizer source the MTP primary serves with
(`--tokenizer=mmangkad/Qwen3.6-27B-NVFP4`) and the only vision-capable 27B; the
`nvidia/Qwen3-32B-NVFP4` dense model also remains a supported candidate — see
`docs/qwen3-32b-nvfp4.md` and `lobes overview --list`.)

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

### Colleague roles: cortex / senses / embedder / reranker / stt / tts

Beyond `cortex`, the **fleet** exposes SIX first-class, Colleague-facing
**roles** (issue #81) — the primary contract callers should address, not raw
model ids: `cortex` (the 27B primary — reasoning/deciding/final authority),
`senses` (the Gemma 4 12B multimodal gear — vision intake/perception; never
decides or takes repo actions; the checkpoint declares audio support but it is
**not currently served** on this vLLM path — issue #101 — so `senses` is
vision-only in practice, and the purpose-built `stt` role, below, is the
supported path for speech), `embedder` (`Qwen/Qwen3-Embedding-0.6B` →
`POST /v1/embeddings`), `reranker` (`Qwen/Qwen3-Reranker-0.6B` → `POST
/v1/rerank` + `/v1/score`), and the opt-in audio overlay's `stt`/`tts`. Roles
are routed by **task family** (`generate` / `embed` / `score` / `rerank`) and
discoverable via `lobes capabilities` / `lobes endpoint <role>` / gateway `GET
/capabilities` — a JSON contract keyed by role (model / runtime / endpoint /
path / context / quant / mtp / responsibilities / forbidden_responsibilities /
ready / loaded); see `docs/colleague-stack.md` for the full contract.
`cortex`/`senses`/`embedder`/`reranker` are default-on and co-reside on the
DGX Spark GB10: `cortex` serves its **full 128K native context at util 0.30**,
`senses` is trimmed to **32K at util 0.14**, and the two ~0.6B pooling gears
run at `*_GPU_MEM_UTIL=0.06` each — default budget `0.30 + 0.14 + 0.06 + 0.06 =
0.56` on the 128 GB GB10. These are the **machine-as-brain** (default)
values — one box hosting every role it can serve; a mesh-brain **deployment
shape** (below) drops one heavy lobe to a peer box and reclaims its budget
instead of merely co-residing it. The 4B `minor` (back-compat `cheap`,
`COMPOSE_PROFILES=minor`, util 0.10) and the legacy 14B Qwen
(`COMPOSE_PROFILES=middle`, util 0.12) are **opt-in** gears and are not
first-class Colleague roles. Callers address the generate lane by
**capability-tier alias** — `model=main|minor|multimodal` (back-compat:
`hard|cheap|normal`), or the Colleague-role names `model=cortex|senses` layered
on top of `main`/`multimodal`; `normal`/`multimodal` maps to the Gemma gear, not
the demoted 14B. A swap/iowait **pressure policy** degrades both `cortex` and
`senses` requests to `minor` (swap > 75 % or iowait > 50 % → degraded, `minor`
only — `senses` is a different capability, not a cheaper rung);
`lobes status --pressure` shows the current tier ceiling. Start/stop one role at
a time with `lobes up <role>` (or the full six-role bundle, `lobes up
colleague-stack`); measure per-role runtime with `lobes measure` and compare
fleet profiles with `lobes benchmark --profile {cortex-only,cortex+senses,
senses-direct,qwen-nvfp4-vs-bf16,all}`. LoRA adapter training targets the 4B
bf16 `minor` only — the 14B NVFP4 is inference-only, and there is no `lobes
train` verb. See `docs/qwen3-embedding-0.6b.md`, `docs/qwen3-reranker-0.6b.md`,
`docs/gemma-4-12b-nvfp4.md`, `docs/gateway-fleet.md`, and
`docs/colleague-stack.md` (the six-role contract).

An opt-in **realtime audio overlay** (`lobes init --fleet --audio`) adds an OpenAI
`/v1/audio/*` facade — a `realtime` bridge container (shipped in the wheel as
`lobes.realtime`) that the gateway fans `/v1/audio/*` out to — backed by two
fixed GPU sidecars: **Parakeet** STT (`nvidia/parakeet-tdt-0.6b-v2`, NeMo ASR →
`POST /v1/audio/transcriptions`) and **Chatterbox** TTS (Resemble AI, 0.5B,
Apache-2.0 → `POST /v1/audio/speech`, 24 kHz, zero-shot voice cloning; it replaced
the retired Magpie NIM — no NGC key). These two are hardcoded, **not** in the
switchable catalog (`lobes/catalog.py`). See `docs/realtime-pipeline.md`,
`docs/parakeet-stt.md`, `docs/chatterbox-tts.md`, and `docs/openai-api.md` (the full
OpenAI-compatible endpoint surface). `lobes explain realtime` / `api` are the
in-CLI versions.

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
card) is the **deployment-shape** axis (issue #113): which of the six
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
or `lobes capabilities` output may claim it validated). Select with `lobes
init --shape <machine-as-brain|spark-lobe|thor-lobe|orin-small>` (dry-run by
default, `--apply` to commit, byte-for-byte restorable by re-running with
the previous shape). A dropped role is flagged `feasible:false` on both
`lobes capabilities` and `GET /capabilities`, omitted from `/v1/models`, and
404s `role_infeasible` on every alias — never half-served. Opt-in **honest
referral** (issue #112, t3): declaring a peer origin per dropped role
(`PRIMARY_PEER_ORIGIN` / `MULTIMODAL_PEER_ORIGIN` / `EMBED_PEER_ORIGIN` /
`RERANK_PEER_ORIGIN` — always operator-typed, never derived, per #92) makes
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
`<PREFIX>_PEER_PROXY` set anywhere — every pre-#115 deployment, and both live
`spark-lobe`/`thor-lobe` boxes as of this writing — every response stays
byte-identical to the pre-proxy contract. See `docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in`
and `docs/deployment-shapes.md#following-the-referral-proxy-lobes-opt-in`.

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
change the world.

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
