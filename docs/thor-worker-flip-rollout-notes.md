# Thor cortex-to-worker flip — rollout note and raw-id consumer audit

**Published BEFORE any flip step runs.** As of this writing (2026-09-10) the
Jetson AGX Thor still serves `cortex` locally
(`unsloth/Qwen3.8-27B-NVFP4` at `max_model_len=262144`, `gpu_mem_util=0.58`,
`WORKER_FEASIBLE=false` — confirmed live from this box's own deployed
`~/.lobes/.env`: `PRIMARY_SERVED_NAME=unsloth/Qwen3.8-27B-NVFP4`,
`VLLM_PORT=8000`, `WORKER_FEASIBLE=false`,
`WORKER_SERVED_NAME=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` — the
last of those is a dormant placeholder, not a live serve). **No flip has
happened yet.** This note exists so every raw-id pinner found below can be
told about the swap ahead of time instead of discovering it as a production
404, mirroring `docs/worker-lightning-rollout-notes.md` (the 2026-08-20
precedent this note is modeled on) and following
`docs/model-switch-playbook.md` §2's rule: **assume every consumer is a
raw-id pinner until the audit proves otherwise.**

## What is flipping

```text
Thor `cortex`  (unsloth/Qwen3.8-27B-NVFP4, 262144, util 0.58)  ->  STOPS being served on Thor
Thor `worker`  (WORKER_FEASIBLE=false today)                    ->  STARTS being served on Thor
```

The new checkpoint is **`nvidia/Qwen3.6-35B-A3B-NVFP4`** — NVIDIA's own
ModelOpt export, distinct from the two Qwen3.6-35B-A3B entries this repo
already carried. Its metadata was read from the checkpoint's own
`config.json` + `hf_quant_config.json` on 2026-09-10 (HF repo last modified
2026-08-29) and committed to `lobes/catalog.py` in the same change series as
this note:

* `Qwen3_5MoeForConditionalGeneration`, `model_type qwen3_5_moe`
* **262144 native**, 256 experts / 8 active, 40 layers
* **MULTIMODAL** — `vision_config` present (deepstack ViT, image + video token ids)
* ModelOpt `MIXED_PRECISION`: experts and shared-expert `W4A16_NVFP4`
  (group_size 16, WEIGHT-only), FP8 on the `linear_attn`/`self_attn`
  projections, `kv_cache_quant_algo: FP8` declared
* `mtp_num_hidden_layers: 1` — it carries its own MTP head — but
  `exclude_modules: ["mtp.layers.0*", "mtp*"]`, i.e. **the MTP module is
  UNQUANTIZED**, which is precisely the condition the 2026-07-31 Thor run
  recorded as marlin's refusal reason ("not supported for unquantized MoE")

The two sibling entries are NOT the target and stay untouched candidates:
`unsloth/Qwen3.6-35B-A3B-NVFP4` (the checkpoint Thor served as `worker`
before deviation d1, `compressed-tensors`, self-hosted MTP — the source of
every historical Thor worker measurement cited below) and
`mmangkad/Qwen3.6-35B-A3B-NVFP4` (32K-native, MTP known not to load).

**Every historical figure in this note and in the per-model doc was measured
on the `unsloth/` export, not on this one.** No box in this fleet has ever
booted `nvidia/Qwen3.6-35B-A3B-NVFP4`; whether it loads at all on the pinned
nightly on sm_110 is the first thing the flip's live spike must answer.

Consequences named in the task brief, restated here so they travel with the
audit:

* every raw-id request for `unsloth/Qwen3.8-27B-NVFP4` that dials the Thor's
  gateway 404s the instant the swap lands;
* the validated Spark+Thor `cortex` replica pool
  (`docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`)
  collapses to single-owner (Spark only);
* the Orin's peer-only `cortex` pool drops from two replicas to one (the
  Spark) — see `docs/evidence/2026-08-30-accept-peer-only-pool-orin.txt`.

## Audit method

Two passes: (1) a filesystem audit of this box (the actual Jetson AGX
Thor) and every sibling checkout physically present under `~/git/`, since
the task brief's `../culture`, `../daria`, `../steward` sibling paths **do
not exist on this box** (verified: `find /home/thor -maxdepth 3 -iname
"culture*" -o -iname "daria*" -o -iname "steward*"` finds no such sibling
checkouts next to `lobes-cli`); and (2) `gh api search/code` against the
`agentculture` org, mirroring the precedent doc's method for the repos this
box doesn't have checked out. Every finding below is cited with file:line
(local) or repo:path (remote); none are inferred.

## Who must change, and what

### 1. The `acp` `vllm-local` provider — THIS box's own deployed agent identity

`culture.yaml:8` (this repo, this checkout):

```yaml
  model: vllm-local/unsloth/Qwen3.8-27B-NVFP4
```

`AGENTS.md:39` repeats the same id in prose (the runtime system prompt
source of truth `culture.yaml`'s `system_prompt` mirrors verbatim,
`culture.yaml:43`). This is the `lobes` agent's own ACP identity — the
literal thing this repo's CLAUDE.md calls "one identity: the gear runs the
model and the agent rides on it." **This is the single most load-bearing
raw-id pin on this box**: the moment Thor stops serving
`unsloth/Qwen3.8-27B-NVFP4`, the `lobes` agent's own ACP provider points at
a dead model id on its own host. `culture.yaml`'s own header comment says
`model:` "must match `VLLM_SERVED_NAME` in the deployment `.env`" and that
`lobes doctor` checks this — so `lobes doctor` should flag the mismatch
post-flip, but the file itself still needs the edit; that edit is **not**
made by this note (task scope is audit + note only).

### 2. The eidetic embed client — NOT broken by this flip, but verified per the brief

`.claude/skills/recall/scripts/recall.sh:169-171` and
`.claude/skills/remember/scripts/remember.sh:160-162` (this repo):

```bash
: "${EIDETIC_EMBED_URL:=http://localhost:8001/v1}"
: "${EIDETIC_EMBED_MODEL:=Qwen/Qwen3-Embedding-0.6B}"
```

This targets the `embedder` role, not `cortex`/`worker` — the flip does not
touch the embed lane, so this consumer is not expected to 404. It is
included here only because the task brief requires it to appear, verified
from the actual scripts (not assumed). **A separate, real mismatch found
while verifying it**: the scripts' hardcoded fallback port is `:8001`, but
this box's actual deployed gateway (`~/.lobes/.env`: `VLLM_PORT=8000`,
confirmed live via `docker ps` — `model-gear-gateway` publishes
`0.0.0.0:8000->8000/tcp`, nothing listens on `:8001`) serves on `:8000`.
Whatever currently makes `recall`/`remember` work on this box must be doing
so via an explicit `EIDETIC_EMBED_URL` override, not the scripts' compiled-in
default — this note does not resolve that discrepancy (out of scope) but
records it so it isn't mistaken for a flip-caused break later.

### 3. `~/.bashrc` — REACHY_* exports on this box (secrets present; values not reproduced)

`/home/thor/.bashrc:20-22` (this box, not this repo — not under version
control):

```bash
export REACHY_OPENAI_API_KEY="<redacted — matches this box's GATEWAY_API_KEY>"
export REACHY_OPENAI_URL_BASE="http://localhost:8001"
export REACHY_OPENAI_MODEL_ID="sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
```

Two things worth recording honestly: `REACHY_OPENAI_URL_BASE` points at
`:8001`, which nothing on this box currently listens on (`ss -tlnp` shows no
`:8001` listener) — so this export looks stale/inert today, independent of
the cortex/worker flip. `REACHY_OPENAI_MODEL_ID` already pins a **third**,
older, demoted candidate (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`), not
`unsloth/Qwen3.8-27B-NVFP4` — so this particular export will not newly break
from *this* flip either way; it appears to already be pointed at something
else, unrelated to cortex or worker. Recorded for completeness, not acted
on (this note doesn't own shell-profile edits).

### 4. `~/.qwen/settings.json` — a live raw-id + raw-URL pin on THIS box

`/home/thor/.qwen/settings.json` (the `qwen-code` CLI's own local config on
this box, not under version control):

```json
"modelProviders": {
  "openai": [
    {
      "id": "unsloth/Qwen3.8-27B-NVFP4",
      "name": "unsloth/Qwen3.8-27B-NVFP4",
      "baseUrl": "http://thor:8000/v1/",
      "envKey": "QWEN_CUSTOM_API_KEY_OPENAI_HTTP_THOR_8000_V1_E352595C28F7"
    }
  ]
},
"model": {
  "name": "unsloth/Qwen3.8-27B-NVFP4",
  "baseUrl": "http://thor:8000/v1/"
}
```

This is a genuine, currently-functional raw-id pin dialing this exact box's
gateway on `:8000` — confirmed reachable (the live gateway). It **will**
404 the moment the flip lands. Not owned by this note to fix (it's a local
tool config, not a repo file), but named here so the operator running the
flip knows to update it. `culture-nodes-agent`'s `qwen_bridge` ACP adapter
(`adapters/qwen/src/qwen_bridge/acp/facts.py:46`) reads `~/.qwen/settings.json`
model identity **by discovery** at session time (`current_model_id` is
"MEASURED from the agent's own responses, never assumed" per that module's
own docstring) — so the bridge itself self-heals once `~/.qwen/settings.json`
is updated; it is `~/.qwen/settings.json` itself that is the actual pin.

### 5. This repo's own vendored `ask-colleague` skill — a fallback default, not this repo's primary consumer

`.claude/skills/ask-colleague/scripts/ask-colleague.sh:194` (this repo):

```bash
MODEL="${COLLEAGUE_MODEL:-${CONVERTIBLE_MODEL:-unsloth/Qwen3.8-27B-NVFP4}}"
```

(`--base-url` on the same script defaults to `http://localhost:8001/v1`,
line 105/193 — the same `:8001` mismatch as finding 2/3 above, independent
of the flip.) This fallback only fires when neither `COLLEAGUE_MODEL` nor
the deprecated `CONVERTIBLE_MODEL` env var is set — `.bashrc` on this box
sets neither, so whatever value is actually in effect for a live
`ask-colleague` invocation here is not visible from static files alone.
The **same vendored copy**, byte-identical, is present in every sibling
checkout under `~/git/` on this box: `jetson`, `microduck-cli`,
`culture-nodes-agent`, `dgx-spark-cli`, `jetson-thor-cli` (grepped
directly, `ask-colleague/scripts/ask-colleague.sh:194` in each) — this is a
mesh-wide vendored skill, not a lobes-cli-specific consumer, and it is the
**upstream** `agentculture/guildmaster` copy of the same file
(`.claude/skills/ask-colleague/scripts/ask-colleague.sh:194`, confirmed via
`gh api search/code`) that all of these were vendored from.

### 6. External repos (via `gh api search/code org:agentculture`) — three live pins found

The precedent doc (`docs/worker-lightning-rollout-notes.md`) found **no**
external raw-id pinner for the `worker` swap. This flip's target id
(`unsloth/Qwen3.8-27B-NVFP4`, the outgoing cortex id) is a different,
older, more deeply wired id — the 2026-08-19 cortex promotion — and this
audit found real, live pins:

* **`agentculture/colleague`, `colleague/config_defaults.py:27`**:

    ```python
    _DEFAULT_MODEL = "unsloth/Qwen3.8-27B-NVFP4"
    ```

  This is the exact field `docs/qwen38-rollout-notes.md` already flagged
  as "the field that *did* break on the cortex swap" for the
  `Qwen3.6-27B-NVFP4 -> Qwen3.8-27B-NVFP4` promotion. It is unrelated to
  which box hosts `cortex` — it breaks whenever `cortex`'s served id
  changes, on any box, which this flip does (Thor stops serving it; the
  Spark keeps serving it under dual-cortex, per the memory record of this
  box's own mesh state, so `_DEFAULT_MODEL` itself does **not** need to
  change for this flip — only clients that were specifically discovering
  Thor's copy do).
* **`agentculture/culture-nodes`, `deploy/prod/pi-developer.json.template:4-5`**:

    ```json
    "model": "unsloth/Qwen3.8-27B-NVFP4",
    "model_endpoint": "http://thor:8000/v1",
    ```

  This is a **Thor-specific** deployment template (`model_endpoint` is
  hardcoded to `http://thor:8000/v1`, not a role-discovered address) — a
  live prod artifact that pins BOTH the raw id and the Thor gateway
  address together. This one is unambiguously in scope for this flip: it
  will 404 the instant Thor stops serving `unsloth/Qwen3.8-27B-NVFP4`,
  regardless of what the Spark still serves.
* **`agentculture/culture-nodes`, `deploy/prod/cutover.sh:161`**:

    ```bash
    MODEL=${CUTOVER_MODEL:-unsloth/Qwen3.8-27B-NVFP4}
    ```

  A prod cutover script default, same repo, same risk as the template
  above — check whether its default is scoped to the Thor endpoint before
  the flip.

Clean (zero hits for `"unsloth/Qwen3.8-27B-NVFP4"` in a targeted
`repo:`-scoped search): `agentculture/embodiment`, `agentculture/eidetic-cli`,
`agentculture/steward`. `agentculture/reachy-mini-cli` and
`agentculture/guildmaster` show hits only in `CHANGELOG.md` and the same
vendored `ask-colleague` skill described in finding 5 — no functional
pin of their own.

`agentculture/culture-nodes-agent`'s own `qwen_bridge` module
(`adapters/qwen/src/qwen_bridge/acp/facts.py:46`, `docs/specs/2026-08-23-qwen-bridge-acp.md:68`,
`docs/plans/2026-08-23-qwen-bridge-acp.md:33`) mentions the id only in
**measured provenance prose** ("model identity is host-local
(spark+thor: unsloth/Qwen3.8-27B-NVFP4, orin: cortex)") from a 2026-08-23
ssh probe of `~/.qwen/settings.json` on all three hosts, and the module's
own docstring insists "Parse, never hard-code" — this is discovery-shaped
code, like `colleague/oilcheck/three_tier.py` in the precedent doc, not a
pin. It will read as historically-accurate-but-stale prose after the flip,
same treatment as the precedent doc gave `embodiment`'s closed experiment
report.

## Operational notes

* **The swap has not happened yet.** As of 2026-09-10, `~/.lobes/.env` on
  this box still reads `PRIMARY_SERVED_NAME=unsloth/Qwen3.8-27B-NVFP4` and
  `WORKER_FEASIBLE=false`.
* **This note does not edit any of the above.** Per task scope, this is
  audit + note only — `culture.yaml`, `~/.qwen/settings.json`,
  `~/.bashrc`, and the external repos above are each a separate operator
  follow-up, timed to land the same day as the flip (the lesson repeated
  verbatim from both precedent notes).
* **Re-run the audit close to the actual flip date.** GitHub code search
  has indexing lag and this snapshot may miss anything added to the mesh
  between now and the live boot.

## Degraded-reasoning note (task step 7)

Two things below the surface of this note are assumption, not
verification, and are flagged here rather than filed anywhere else, per
the task's own instruction:

1. **RESOLVED — the checkpoint identity is confirmed.** An earlier draft of
   this note could not find `nvidia/Qwen3.6-35B-A3B-NVFP4` in the catalog
   and resolved the flip target to `unsloth/Qwen3.6-35B-A3B-NVFP4` instead,
   flagging the inference rather than hiding it. That inference was WRONG
   and has been corrected throughout: the id is real (HF repo last modified
   2026-08-29), its metadata was read directly from the checkpoint on
   2026-09-10, and it is now a first-class catalog entry holding
   `role_hint="worker"`. The reason the audit could not find it is simply
   that the catalog entry did not exist yet when the audit ran. Recorded
   here rather than quietly edited away, because the checkpoint-identity
   question is exactly the kind a rollout note exists to settle.

2. **Power mode / L4T / clocks were read live on 2026-09-10** (`nvpmodel
   -q` → `MAXN`; `/etc/nv_tegra_release` → `R38 (release), REVISION: 2.2`)
   for the per-model doc addition below, because no prior evidence
   transcript for this checkpoint on Thor recorded them
   (`docs/evidence/2026-07-31-accept-worker-thor.txt` and
   `docs/evidence/2026-08-20-baseline-worker-qwen35b-thor.txt` both omit
   power mode/clocks entirely). These are **today's idle-box values**, not
   necessarily what was in effect during either historical measurement —
   they are recorded as current-state context for an operator re-running
   the recipe, not as a re-statement of the historical runs' conditions.

## See also

* `docs/worker-lightning-rollout-notes.md` — the precedent this note is
  modeled on (same shape, `worker`'s prior id swap).
* `docs/qwen38-rollout-notes.md` — the original cortex-swap audit; this
  note extends its finding (`colleague/config_defaults.py`'s
  `_DEFAULT_MODEL`) with two newly-found `culture-nodes` prod pins.
* `docs/model-switch-playbook.md` §2 — the general playbook both audits
  follow.
* `docs/qwen3.6-35b-a3b-nvfp4.md` — the per-model doc extended (see below)
  with the re-run recipe for `unsloth/Qwen3.6-35B-A3B-NVFP4` reclaiming the
  Thor `worker` seat.
* `docs/evidence/2026-07-31-accept-worker-thor.txt`,
  `docs/evidence/2026-08-20-baseline-worker-qwen35b-thor.txt` — the two
  transcripts this checkpoint's Thor history is built from.
