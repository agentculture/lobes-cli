# Qwen3.6-35B-A3B-NVFP4: two checkpoints, two stories

The catalog carries **two, distinct** `Qwen3.6-35B-A3B-NVFP4` entries — same
architecture family (MoE, ~35B total / ~3B active per token), different
org/export, different role, different story:

- **`unsloth/Qwen3.6-35B-A3B-NVFP4`** — the **former `worker`** role holder
  (thor-worker-lobe plan; DEMOTED to candidate 2026-08-20, deviation d1).
  MULTIMODAL, ships its OWN self-hosted MTP draft, 262144 native context. See
  ["`worker`: DEMOTED from the eighth Colleague
  role"](#worker-demoted-from-the-eighth-colleague-role-2026-08-20-unslothqwen36-35b-a3b-nvfp4)
  below.
- **`mmangkad/Qwen3.6-35B-A3B-NVFP4`** — a **MoE candidate**, the *former*
  fleet fallback, 32K native, its own MTP explicitly does not load. Unchanged
  by the `worker` role's addition — see the rest of this document below.

The two are deliberately kept as separate catalog entries (never merged):
they resolve to different `role_hint`s, different native context windows,
different quantization conventions, and — critically — one (`mmangkad/`)
has an MTP config that is *known* not to load, while the other
(`unsloth/`) ships its own MTP draft module whose loadability is genuinely
unconfirmed, not assumed working. Treating them as interchangeable would be
exactly the kind of card-prose-over-measurement mistake this repo's honesty
rules exist to prevent.

## `worker`: DEMOTED from the eighth Colleague role, 2026-08-20 (`unsloth/Qwen3.6-35B-A3B-NVFP4`)

> **Status: no longer `worker`.** Deviation d1 (2026-08-20,
> `.devague/deliveries/nemotron-lightning-worker.json`) replaced this
> checkpoint in the `worker` seat with
> `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4`, now hosted on the
> DGX Spark GB10 — see
> [`nemotron-3.5-lightning-30b-a3b-nvfp4.md`](nemotron-3.5-lightning-30b-a3b-nvfp4.md).
> This checkpoint is demoted to a kept catalog **candidate**
> (cite-don't-delete), not deleted — the section below is now history, not
> the current `worker` contract.
>
> **Final production baseline, captured 2026-08-20 just before the swap**
> (`docs/evidence/2026-08-20-baseline-worker-qwen35b-thor.txt`, Jetson AGX
> Thor, production 0.23.1 engine, `util=0.45`, `max_model_len=262144`, MTP
> self-draft ON): known-answer "Paris" PASS (5.43 s), decode **61.2 tok/s**
> single-stream (679 completion tokens / 11.1 s), structured tool-call PASS
> (4.92 s) — this number is unrecoverable after the flip and is the figure
> the Lightning replacement is compared against.
>
> **Its own GDN-MTP kernel gap, on the fleet's newer nightly.** Before the
> baseline above, an attempt to re-boot this checkpoint on the fleet-wide
> `8bd082` nightly (vLLM 0.26.1rc1.dev942, the same digest the Lightning
> spike later ran on) BOOTED healthy — KV pool grew to 50.02 GiB /
> 4,317,665 tokens / 16.47× — but DIED on the first decode request:
> `RuntimeError: launch_gdn_decode_post_conv_mtp ... GDN decode MTP
> post-conv kernel launch failed: no kernel image is available for
> execution on the device`. Torch-level sm_110 SASS is present, but this
> digest's own csrc GDN (Mamba/gated-delta-net) MTP decode kernel ships no
> sm_110 image — the same kernel-coverage gap the `cortex` swap
> independently hit on this checkpoint's sibling architecture (see
> [`docs/machine-profiles.md`](machine-profiles.md) and
> `docs/evidence/2026-08-20-accept-cortex-local-thor.txt`). The baseline was
> therefore captured on the production 0.23.1 engine instead, which is also
> what the covering plan's playbook asked for.

The section below is preserved as-written for its historical DECLARED
contract (2026-07-31, before the swap):

> **Status (historical, pre-d1): DECLARED, not yet booted on any hardware.**
> The catalog entry, role registry, and gateway config wiring were
> **shipped** (verified against the checkpoint's own `config.json`, fetched
> 2026-07-31 — not card prose). The `thor-worker` deployment shape, its
> compose service, and every live-measured number (`gpu_mem_util`,
> `max_model_len` if trimmed from native, the sm_110 MoE backend choice, MTP
> acceptance) were **forthcoming** (thor-worker-lobe plan task t7) —
> nothing claimed worker validated on hardware (#108). d1 superseded this
> before t7 landed: `thor-worker`'s shape data ended up hosting Lightning on
> the Spark card instead (see
> [`deployment-shapes.md`](deployment-shapes.md#shapes-are-card-agnostic-data-proven-live-by-d1)).

**Model id:** `unsloth/Qwen3.6-35B-A3B-NVFP4`
**Tier alias:** `worker` — like `muse`, the role name *is* the alias
(capability order: `minor` < `multimodal` < `worker` < `muse` < `primary`/`main`).
**Role:** `worker` — the fleet's fast **ground-work DOER**, the EIGHTH
first-class Colleague role. **Opt-in for hosting**: `machine-as-brain` never
hosts it; only an explicit worker-hosting shape (`thor-worker`, forthcoming)
will.
**Status:** `configured` in the catalog (declared 2026-07-31; not yet booted
on any hardware — task t7 gates that).

### What it is

Qwen3.6 35B-A3B (a DISTINCT export from the `mmangkad/` candidate below —
same architecture family, different org). Facts verified against the
checkpoint's actual `config.json` + the absence of a separate
`hf_quant_config.json` (fetched 2026-07-31, not card prose):

- **MoE, ~3B active parameters per token** — `architectures:
  ["Qwen3_5MoeForConditionalGeneration"]`, `model_type: "qwen3_5_moe"`,
  `num_experts=256`, `num_experts_per_tok=8`. vLLM loads *all* experts into
  memory; the small active set only reduces per-token compute (the same MoE
  decode-speed advantage the `mmangkad/` sibling demonstrated live, below).
- **262144 native context** (`text_config.max_position_embeddings`), the
  card additionally advertising a YaRN-extended ~1.01M window (unconfirmed
  here — the catalog carries the native figure only).
- **Ships its OWN self-hosted MTP draft module** — unlike the `mmangkad/`
  candidate, whose MTP explicitly fails to load. `text_config
  .mtp_num_hidden_layers=1`, and `quantization_config.ignore` carries a
  `"re:^mtp.*"` pattern — i.e. the checkpoint's own MTP weight tensors
  physically exist and are deliberately left UNQUANTIZED, confirming the
  self-hosted draft the card describes ("can act as its own speculative
  draft for faster decoding"). The README's own vLLM MTP serve command
  matches: `--speculative-config '{"method": "mtp",
  "num_speculative_tokens": 2}'` — no external `model`/`draft_model_id` key,
  because the draft lives IN this checkpoint. **Loadability on the deployed
  vLLM image and MTP's acceptance rate are UNCONFIRMED until the live boot
  (task t7)** — the `mmangkad/` sibling's own MTP attempt failed with a
  weight-shape mismatch on a *different* checkpoint's draft, so this is a
  genuinely open question, not a formality.
- **`compressed-tensors` quantization** — `quantization_config.quant_method
  ="compressed-tensors"` (mixed precision: 8-bit float-quantized
  attention/lm_head/upper MLP layers, 4-bit nvfp4-pack-quantized MoE
  experts) — NOT nvidia `modelopt`, unlike the `mmangkad/` candidate. No
  separate `hf_quant_config.json` exists (a `compressed-tensors` checkpoint
  carries its quant config inline in `config.json`; that separate file is a
  modelopt/TensorRT export convention this checkpoint doesn't use).
- **MULTIMODAL — image+video, no audio.** `config.json` carries a
  `vision_config` (27-layer ViT), `image_token_id`/`video_token_id`, and
  vision start/end tokens, and **no** `audio_config`. **Operator decision
  (2026-07-31): worker is served MULTIMODAL** — a "seeing doer" (image+video
  intake + `repo_action`) — so the compose lane will NOT pass
  `--language-model-only` (unlike the 27B `cortex` MTP primary, whose export
  dropped its ViT). Whether vLLM actually serves
  `Qwen3_5MoeForConditionalGeneration` + MTP together on Thor's sm_110 is
  **UNCONFIRMED until the live boot** (task t7).
- **`qwen3_coder` tool-call parser** — the same Qwen-family parser pair
  `cortex` uses (`--tool-call-parser=qwen3_coder` **plus**
  `--reasoning-parser=qwen3`), never inferred from the model card — see
  ["vLLM parser pairs are per-family"](#tool-calling-the-qwen-family-parser-pair)
  below.
- **The README's DGX Spark serving note recommends `--moe-backend
  flashinfer_b12x` under `CUTE_DSL_ARCH=sm_121a`** — explicitly *against*
  `marlin` ("2x slower") on that arch. `sm_121a` is the **Spark's** arch, not
  Thor's (**sm_110** — see [`docs/machine-profiles.md`](machine-profiles.md)
  and the CUDA-wheel-arch-is-not-a-family lesson: Spark and Thor are
  different SASS targets even though both are "Blackwell-class"). The
  catalog carries `flashinfer_b12x` as the best-cited default, but the
  correct sm_110 MoE backend for Thor is **UNCONFIRMED** until task t7's
  live boot chooses (or refuses) it — the `mmangkad/` sibling's own
  sm_110-vs-sm_121 story (below) is exactly why this isn't assumed.

### Responsibilities: the fast ground-work DOER, and the first non-`cortex` actor

`worker`'s responsibilities: `execution`, `ground_work`, `bulk_transform`,
`drafting`, `image_understanding`, `video_understanding`, `tool_use`, and —
uniquely among every role besides `cortex` — **`repo_action`**. Forbidden:
`final_decision`, `security_decision`. `worker` executes bulk ground work
(drafting, transforms, image/video-informed edits) UNDER `cortex`'s
direction; it never makes the final call or a security decision on its own
authority. This is a materially different contract from `senses` (perceives,
never acts) and `muse` (proposes via tool calls, never acts) — see
[`docs/colleague-stack.md`](colleague-stack.md) for the full division of
labour across all nine roles.

### Tool calling: the Qwen-family parser pair

`worker` is specified to serve tool calls on the same **matched pair** the
`cortex` lane has always used, never a parser inferred from the model card
(the recorded, hard-won lesson from the Gemma 4 tool-calling incident — see
[`docs/gemma-4-31b-nvfp4.md`](gemma-4-31b-nvfp4.md#tool-calling) and
`CLAUDE.md`'s "Gemma 4 tool calling" section):

```text
--tool-call-parser=qwen3_coder     # the same parser cortex uses
--reasoning-parser=qwen3           # the same reasoning parser cortex uses
```

This is a Qwen3.6 checkpoint, and the catalog's `infer_parser` already
resolves the unsloth id to `qwen3_coder` (mirroring the `mmangkad/`
candidate's own `qwen3_coder` entry) — but per the per-family parser rule,
**this must still be verified live** with a `skip_special_tokens:false`
probe against the actually-served worker lane before it is trusted, exactly
as the Gemma 4 pair had to be (task t7/t9 — a live, evidence-backed
verification, not a repeat of a never-validated guess).

### How it will be hosted: the `thor-worker` shape (forthcoming)

`worker` is the **second opt-in core role**
(`lobes/profiles/shapes.py`'s `OPT_IN_CORE_ROLES = ("muse", "worker")`),
mirroring `muse`'s mechanics exactly: never hosted by `machine-as-brain`, the
gateway wires its backend only behind `WORKER_BASE_URL`, an unwired `worker`
defaults to infeasible (`model=worker` 404s `role_infeasible`, never a
silent fallback), and `base.toml` vetoes it on an unrecognised card just
like `muse`. The shape/gateway-config machinery that would host it is
**shipped** (`OPT_IN_CORE_ROLES`, `WORKER_FEASIBLE`/`WORKER_PEER_ORIGIN`/
`WORKER_PEER_PROXY`/`WORKER_PEER_API_KEY`, `shape_render.py`'s
`OPT_IN_CORE_ACTIVATION_ENV["worker"]`); what remains is the `thor-worker`
shape's own TOML (with a live-measured `[overrides.worker]` budget,
mirroring `thor-muse`'s `0.40→0.55` measured-not-arithmetic pattern), the
`vllm-worker` compose service, and the CLI verb polish (`lobes up worker`) —
all forthcoming, thor-worker-lobe plan tasks t4/t6/t7. **No budget number is
declared here** — it is committed only once a live boot on the physical
Jetson AGX Thor produces it.

```bash
lobes init --shape thor-worker --apply   # forthcoming — not yet a valid shape name
lobes fleet up --apply
lobes up worker --apply                  # verb wired (mirrors `lobes up muse`); needs a worker-hosting shape to actually boot (thor-worker, t7)
```

`thor-worker` will drop BOTH heavy default lobes (`cortex` and `senses`) to
peer boxes, exactly like `thor-muse` does today — the physical Thor that
previously hosted `thor-muse` is the box this shape targets. See
[`docs/deployment-shapes.md`](deployment-shapes.md) for the opt-in-core-role
mechanism shared with `muse`, and `CLAUDE.md`'s "Colleague roles" section for
the mesh-wide picture (including `muse`'s new DORMANT status on this same
box).

### Related docs

- [`docs/colleague-stack.md`](colleague-stack.md) — the nine-role Colleague
  contract, `worker`'s `responsibilities`/`forbidden_responsibilities`
  exactly as declared, and the "first non-`cortex` actor" division of labour.
- [`docs/deployment-shapes.md`](deployment-shapes.md) — the opt-in-core-role
  concept `worker` shares with `muse`, and the `thor-worker` shape's status.
- [`docs/gateway-fleet.md`](gateway-fleet.md) — the `worker` tier alias, the
  inverted feasibility default, peer channels, pressure policy.
- [`docs/gemma-4-31b-nvfp4.md`](gemma-4-31b-nvfp4.md) — `muse`, the sibling
  opt-in core role, now DORMANT on the box `worker` is moving onto.
- `CLAUDE.md`'s "Colleague roles" section — the nine-role summary and the
  muse-to-worker mesh migration in one place.

---

## `nvidia/Qwen3.6-35B-A3B-NVFP4` — MOVED to its own doc

> **This checkpoint is now VALIDATED and deployed as the Thor's `worker`.** Its
> full recipe, docker/compose setup, measured budgets, speculation sweep and
> operational traps live in
> **[`nvidia-qwen3.6-35b-a3b-nvfp4.md`](nvidia-qwen3.6-35b-a3b-nvfp4.md)**.
> The section below is the pre-boot summary written before it was measured;
> the numbers that matter are in the dedicated doc.

### (pre-boot summary, superseded)

> **Status: DECLARED, never booted.** This is the checkpoint issue #244 puts
> on Thor's `worker` seat. It is NVIDIA's own ModelOpt export and is a
> DIFFERENT checkpoint from the `unsloth/` and `mmangkad/` siblings this doc
> otherwise covers. **No box in this fleet has ever loaded it**, so this
> section contains no throughput, no acceptance rate, and no budget — only
> what the checkpoint's own files say.

Read from the checkpoint's `config.json` + `hf_quant_config.json` on
2026-09-10 (HF repo last modified 2026-08-29):

| field | value |
|---|---|
| architecture | `Qwen3_5MoeForConditionalGeneration` (`qwen3_5_moe`) |
| native context | **262144** |
| experts | 256 total / 8 active, 40 layers |
| modality | **MULTIMODAL** — `vision_config` present (deepstack ViT, image + video token ids) |
| quantization | ModelOpt `MIXED_PRECISION`: experts + shared-expert `W4A16_NVFP4` (group_size 16, **weight-only**), FP8 on `linear_attn`/`self_attn` projections |
| KV cache | `kv_cache_quant_algo: FP8` **declared** |
| MTP | `mtp_num_hidden_layers: 1` — carries its own head — but `exclude_modules: ["mtp.layers.0*", "mtp*"]`, so **the MTP module is UNQUANTIZED** |

That last row matters more than it looks. The 2026-07-31 Thor run recorded
`marlin` failing with *"not supported for unquantized MoE — the self-hosted
MTP experts are unquantized"*. This export has the same property, so the
issue's headline `--moe-backend marlin` recommendation is a **hypothesis to
test live, not a default to copy**. Note the counter-consideration: unlike
the `unsloth/` export, this one's MAIN experts are `W4A16_NVFP4`
(weight-only) — the same family that let Marlin work for Lightning on the
Orin's sm_87 — so the arms genuinely have to be run rather than predicted.

**What must be measured before anything here is claimed** (plan
`docs/plans/2026-09-10-thor-worker-arm-qwen3-6-35b-a3b-recipes.md`, task
t11): whether it loads at all on the pinned nightly on sm_110; which MoE
backend boots; the gpu_mem_util / max_model_len budget; the parser pair
verified live with `skip_special_tokens: false`; and image + video intake
with a negative control.

See `docs/thor-worker-flip-rollout-notes.md` for the raw-id consumer audit
that must be published before the flip.

### Prior art — this repo already failed to load this exact id once

Read *"Why we serve the `mmangkad/` copy, not `nvidia/`"* further down this
doc before running the spike. On **2026-05-31**, on the DGX Spark GB10, this
same `nvidia/Qwen3.6-35B-A3B-NVFP4` id **would not load**: on vLLM 0.19.0 and
0.21.0 the MoE expert loader failed with `KeyError:
layers.0.mlp.experts.w2_input_scale` on `triton`/auto and *"not supported for
unquantized MoE"* on `marlin`, under both `--quantization modelopt` and
`modelopt_fp4`. That is the recorded origin of risk r1 in the #244 plan.

Two things have changed since, and neither is proof either way:

* the engine moved from 0.19/0.21 to the pinned `8bd082` nightly
  (`0.26.1rc1.dev942`), which is four minor versions of NVFP4-MoE loader work
  later; and
* the HF repo itself was **re-uploaded 2026-08-29**, so the export on the hub
  today is not necessarily the one that failed in May.

The spike therefore has to actually run. If it fails the same way, that is a
publishable result and the fallback recipe below is the answer — not a reason
to quietly substitute a different checkpoint.



## `unsloth/Qwen3.6-35B-A3B-NVFP4` re-taking Thor's `worker` seat — the FALLBACK recipe

> **This section is the ROLLBACK/fallback path, not the #244 target.** It
> describes re-promoting the demoted `unsloth/` candidate — the checkpoint
> Thor actually served as `worker` before deviation d1 — and every figure in
> it was measured on THAT export. It is kept because if the `nvidia/` target
> above fails to load on sm_110, this is the known-good recipe to fall back
> to.

> **Status: UNMEASURED for this specific re-run.** A flip is proposed that
> stops Thor's `cortex` (`unsloth/Qwen3.8-27B-NVFP4`) and re-promotes this
> checkpoint back into Thor's `worker` seat — reversing the demotion above.
> See `docs/thor-worker-flip-rollout-notes.md` for the raw-id consumer
> audit. **Every number below either cites an
> existing evidence transcript from this checkpoint's PRIOR life as Thor's
> `worker` (2026-07-31/2026-08-20, before deviation d1), or is marked
> NOT YET MEASURED.** None of it is a claim about a post-flip boot that has
> not happened — the #108 honesty rule applies here exactly as everywhere
> else in this repo.

### Re-run recipe (everything an operator needs, in one place)

**Container image (engine pin) — this is the load-bearing gotcha:**

| Pin | Digest | vLLM | Status for THIS checkpoint |
|---|---|---|---|
| Production pin (2026-07-31/2026-08-20 evidence) | `vllm/vllm-openai@sha256:7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9` | `0.23.1rc1.dev672` | **VALIDATED** — both cited transcripts ran on this pin; MTP self-draft ON measured 89.1% acceptance, 50.8-61.2 tok/s single-stream. |
| Current fleet-wide default (`VLLM_NIGHTLY_IMAGE`, `lobes/templates/fleet/env.example`) | `vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695` | `0.26.1rc1.dev942` | **KNOWN BROKEN with MTP on** — the 2026-08-20 baseline's own "FAILED PRELUDE" section recreated this exact checkpoint on this exact digest: it booted healthy (KV pool 50.02 GiB / 4,317,665 tokens / 16.47x) but **died on the first decode request** with `RuntimeError: launch_gdn_decode_post_conv_mtp ... no kernel image is available for execution on the device` — this digest's csrc GDN/MTP decode kernel ships no sm_110 image. Plain (non-MTP) decode on this digest is **NOT YET MEASURED** for this checkpoint. |

**Recommendation for a re-run:** pin `WORKER_IMAGE` explicitly to the
7c5a10e9... digest if MTP self-draft is wanted (it is what the cited
numbers were measured on); if using the current fleet default nightly
instead, leave `WORKER_SPECULATIVE_CONFIG` unset (its template default) and
treat plain-decode throughput on that nightly as unmeasured until proven.

**Exact argv** (from `lobes/templates/fleet/docker-compose.yml`'s
`vllm-worker` service, substituting the overrides this checkpoint needs —
per that file's own inline comment: "If you serve the demoted
`unsloth/Qwen3.6-35B-A3B-NVFP4` candidate here instead, override
`WORKER_QUANTIZATION=compressed-tensors` alongside `WORKER_MODEL`" and
"override `WORKER_REASONING_PARSER=qwen3` to match it"):

```bash
vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name=unsloth/Qwen3.6-35B-A3B-NVFP4 \
  --host=0.0.0.0 \
  --port=8000 \
  --quantization=compressed-tensors \
  --max-model-len=262144 \
  --gpu-memory-utilization=0.45 \
  --enable-auto-tool-choice \
  --tool-call-parser=qwen3_coder \
  --reasoning-parser=qwen3 \
  --trust-remote-code
  # add, only if pinning the 7c5a10e9... image and wanting MTP:
  # --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}'
```

`.env` overrides to set (`lobes/profiles/builtin_shapes/thor-worker.toml`'s
`[overrides.worker]` block, unchanged since 2026-07-31):

```bash
WORKER_MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4
WORKER_SERVED_NAME=unsloth/Qwen3.6-35B-A3B-NVFP4
WORKER_QUANTIZATION=compressed-tensors
WORKER_MAX_MODEL_LEN=262144
WORKER_GPU_MEM_UTIL=0.45
WORKER_REASONING_PARSER=qwen3
WORKER_FEASIBLE=true
COMPOSE_PROFILES=worker   # un-gates the profile-gated vllm-worker service
```

**MoE backend: do NOT force one.** Measured live on this exact box
(2026-07-31): `flashinfer_b12x`/`flashinfer_cutlass` FAIL (sm_121a/Spark-only
kernels), `marlin` FAILS ("not supported for unquantized MoE" — the
self-hosted MTP experts are unquantized), `triton` FAILS ("not supported for
NvFP4 MoE"). Only **auto-select** (omit `--moe-backend` entirely) boots —
vLLM picks TRITON for the unquantized/fp8 MoE and a modular NVFP4 kernel for
the main experts.

**JetPack / L4T, power mode, clocks:** **NOT RECORDED in either cited
evidence transcript** (`docs/evidence/2026-07-31-accept-worker-thor.txt`,
`docs/evidence/2026-08-20-baseline-worker-qwen35b-thor.txt` both omit these
fields entirely — a gap in those transcripts, not something this doc can
retroactively fill). Read live from this box on 2026-09-10, while writing
this note, for context only — **not proof of the conditions either cited
transcript ran under**:

```text
$ nvpmodel -q
NV Power Mode: MAXN

$ cat /etc/nv_tegra_release
# R38 (release), REVISION: 2.2, GCID: 42205042, BOARD: generic, EABI: aarch64, DATE: Thu Sep 25 22:47:11 UTC 2025
```

A re-run should capture `nvpmodel -q` and
`/sys/class/devfreq/17000000.gpu/{cur,min,max}_freq` alongside its own
throughput numbers, per `docs/measuring-lane-performance.md` Rule 3 — do
not assume MAXN / R38 2.2 were in effect for the cited historical numbers
just because they're in effect today.

**Co-resident set** (`lobes/profiles/builtin_shapes/thor-worker.toml`,
`hosts = ["worker", "hand", "embedder", "reranker", "stt", "tts"]`):
`worker` + `hand` (LFM2.5-1.2B, per-card util) + `embedder` (util 0.06) +
`reranker` (util 0.06) + the opt-in audio overlay if enabled — **no
`cortex`, no `senses`**. `muse` stays dormant/unhosted. This is the
inverse of today's live shape (cortex + hand + embedder + reranker, no
worker) — the flip is a shape change, not just a served-id change.

**Known-good throughput (from the checkpoint's PRIOR Thor life, NOT
re-validated for this re-run):**

| Metric | Value | Source |
|---|---|---|
| Model load | 24.81 GiB / ~31 s | `2026-07-31-accept-worker-thor.txt` |
| KV cache pool | 41.78 GiB = 14.07x ceiling at 262,144 tokens/request | same |
| Decode (with thinking) | 50.8 tok/s | same |
| Decode (no thinking, sustained) | 73.5 tok/s (600 tok / 8.17s) | same |
| Decode (production re-baseline, 2026-08-20, just before the swap to Lightning) | 61.2 tok/s (679 tok / 11.1s) | `2026-08-20-baseline-worker-qwen35b-thor.txt` |
| MTP self-draft acceptance | 89.1% (385/432) | `2026-07-31-accept-worker-thor.txt` |
| TTFT (short prompt) | 2102 ms | same |
| Vision (red/blue image + negative control) | PASS | same |
| Video (real webcam clip, 78 KB) | PASS (accurate scene/subject/motion) | same |

**! 14.07x is a KV-pool ceiling, not measured concurrency** — the same
shape-file warning applies here: usable concurrency saturates near width
8-9 in independent measurements, not 14 (see
`lobes/profiles/builtin_shapes/thor-worker.toml`'s own warning block).

**What is genuinely NOT YET MEASURED for a re-run today:**
- plain (non-MTP) decode throughput on the current fleet-default nightly
  digest (`8bd082...`) — only the MTP-on path was tried on that digest, and
  it crashed;
- any number at all under Thor's *current* clocks/power state as opposed to
  whatever was in effect 2026-07-31/2026-08-20;
- co-residency effects with `hand`/`embedder`/`reranker`/audio all loaded
  simultaneously (the cited transcripts were single-service boots against
  this checkpoint, per the thor-worker shape's own resource layout).

---

## MoE candidate: `mmangkad/Qwen3.6-35B-A3B-NVFP4`

A **MoE candidate** — the *former* fleet fallback. It was **superseded as the
fallback choice** by the dense `RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4`
([`docs/mistral-small-3.2-24b-nvfp4.md`](mistral-small-3.2-24b-nvfp4.md)), because
this checkpoint never loaded on the GB10 (see the status note below). (The fleet
now runs one *generate* backend by default — any warm fallback is opt-in via the
`FALLBACK_*` keys.) It remains
in the **supported catalog** as a candidate to re-test on a quiet/dedicated box
(`lobes overview --list`). See [`docs/gateway-fleet.md`](gateway-fleet.md) for the
fleet topology and the
[catalog-vs-warm distinction](gateway-fleet.md#supported-catalog-vs-warm-backends)
(what you *can* load vs. what's loaded *now*).

Source: <https://huggingface.co/mmangkad/Qwen3.6-35B-A3B-NVFP4>.

**This entry is unchanged by the `worker` role's addition above** — no
promotion, no removal, no rewrite (thor-worker-lobe plan non-goal). It stays
exactly the candidate it was.

> **Status: load-tested 2026-05-30 — does NOT load reliably on this GB10.** First
> live `lobes fleet up` on `spark-f8a9`: co-resident with the 27B primary it hit
> `CUDA error: out of memory` on engine init and crash-looped (14+ restarts);
> *solo* (65 GiB free) it still crashed/restarted and then stalled at "Loading
> safetensors checkpoint shards: 0%" with the GPU idle, never reaching `/health`
> in 8+ min. **No benchmark obtained.** The architecture-derived expectations
> below are *unconfirmed*. Two root causes are entangled and need separating:
> (1) co-residence with another ~30B model overruns the 121.7 GiB unified pool
> (see [`docs/gateway-fleet.md`](gateway-fleet.md)); (2) the checkpoint's own
> load path (MoE + multimodal ViT + Mamba, single 24 GiB safetensors) stalls/OOMs
> even solo under swap pressure. Re-test on a quiet box before relying on it.

**Update — load-tested 2026-05-31 — DOES load solo with the right flags.** With
the 27B primary stopped (so the 35B had the GB10 to itself) and shahizat's tuning
(`--moe-backend marlin`, flashinfer, async scheduling, chunked prefill) at
`--gpu-memory-utilization 0.70 --max-model-len 32768`, it loaded healthy in ~6 min
(~84 GiB resident) and served. Two caveats found: (1) `0.85` util fails the
pre-flight reservation on this *shared* box (only ~90 of 121.7 GiB free — the
audio NIMs + reachy hold the rest), so `0.70` is the working value; (2) the MTP
`--speculative-config` from shahizat's recipe **fails to load** on this `mmangkad/`
copy (`qwen3_5_mtp.py` weight-shape mismatch on vLLM nv26.04) — it is tied to his
`nvidia/` checkpoint. Measured numbers under "Live replication" below.

## What it is

- An **NVFP4 (Mixture-of-Experts)** checkpoint: ~35B total parameters, **~3B
  active per token** (`A3B`). vLLM loads *all* experts into memory; the small
  active set only reduces per-token compute.
- Decode is memory-bandwidth bound on the GB10 (~273 GB/s shared). Reading only
  ~3B active params per token (≈1.5 GB at 4-bit) gives an **expected decode
  ceiling far above the dense 32B** (which reads ~18 GB/token) — the reason it is
  the fast fallback. *Confirm live.*

## How it runs in the fleet

Configured via the `FALLBACK_*` keys in the fleet `.env` (scaffolded by
`lobes init --fleet`); served by the `model-gear-vllm-fallback` container:

```dotenv
FALLBACK_MODEL=mmangkad/Qwen3.6-35B-A3B-NVFP4
FALLBACK_SERVED_NAME=mmangkad/Qwen3.6-35B-A3B-NVFP4
FALLBACK_MAX_MODEL_LEN=32768
FALLBACK_GPU_MEM_UTIL=0.35          # both models warm: keep primary+fallback well under 1.0 (dedicated box)
FALLBACK_TOOL_CALL_PARSER=qwen3_coder
FALLBACK_QUANTIZATION=modelopt_fp4
```

Address it through the gateway by name (or set `GATEWAY_ALIASES` for a short
alias):

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -d '{"model":"mmangkad/Qwen3.6-35B-A3B-NVFP4","messages":[{"role":"user","content":"hi"}]}'
```

## Caveats to confirm on first load

1. **Tool-call format.** Qwen3.6 emits the Qwen3-Coder **XML** function format, so
   the backend is served with `--tool-call-parser=qwen3_coder` (not the `hermes`
   parser the dense Qwen3-32B uses). `lobes.runtime._parser.infer_parser`
   already maps `qwen3.6` → `qwen3_coder`. Verify a `tool_choice:"auto"` probe
   returns a `finish` tool call.
2. **Quantization format.** The fleet defaults `FALLBACK_QUANTIZATION=modelopt_fp4`
   (as for the `nvidia/` checkpoints). This community (`mmangkad`) checkpoint may
   instead be a compressed-tensors NVFP4 — if vLLM rejects `modelopt_fp4`, drop or
   change `FALLBACK_QUANTIZATION`.
3. **`--trust-remote-code`.** The fleet compose omits it (as the single-model
   template does). If this checkpoint ships custom modeling code, vLLM will say so
   on load; add it back deliberately (it lets repo code run in-container alongside
   `HF_TOKEN` and the mounted cache).
4. **Architecture support.** Confirm the engine registers the checkpoint's
   architecture, as done for the 27B sibling:
   `docker exec model-gear-vllm-fallback python3 -c "from
   vllm.model_executor.models.registry import ModelRegistry;
   print(ModelRegistry.get_supported_archs())"`.

## Benchmark — blocked (model would not load), 2026-05-30

A live run was attempted (`lobes fleet up --apply` on `spark-f8a9`, then
`lobes benchmark --model mmangkad/Qwen3.6-35B-A3B-NVFP4`). The model never reached
`/health`, so no numbers exist yet:

| Property | Value |
|---|---|
| Health / `max_model_len` | **never healthy** — crash-looped co-resident; stalled at safetensors 0 % solo |
| Weights on disk | 24 GiB (single `model.safetensors`; `Qwen3_5MoeForConditionalGeneration`) |
| Decode throughput | *blocked* — `lobes benchmark` returned HTTP 502 (backend not up) |
| Prefill / correctness / tool calling | *blocked* |
| Co-resident with 27B (util 0.55/0.30, then 0.40/0.35) | **OOM** — `CUDA error: out of memory` on engine init |
| Solo (util 0.30, 65 GiB free) | crashed/restarted, then stalled loading the 24 GiB shard with GPU idle |

Next: re-test on a **dedicated/quiet** GB10 (stop other GPU services first), and
isolate whether the failure is co-residence pressure or the checkpoint's own
load path. Consider `--enforce-eager` (skip CUDA-graph capture) and disabling
`--enable-prefix-caching` to shrink the warmup footprint on the first load.

## Reference serve recipe + benchmark (shahizat, dedicated boxes)

shahizat benchmarked this model — the **`nvidia/Qwen3.6-35B-A3B-NVFP4`** checkpoint
(a different repo from the catalogued `mmangkad/` copy above) — on dedicated DGX
Spark, Jetson Thor, and Blackwell 6000 Pro boxes, where it **did** load and serve:
[NVIDIA Developer Forums, 2026-05-31](https://forums.developer.nvidia.com/t/benchmark-report-qwen3-6-35b-a3b-nvfp4-on-nvidia-dgx-spark-jetson-thor-blackwell-6000-pro/371810).
This is the serve recipe to try when re-testing on a quiet box. The two
**MoE-only** flags (`--moe-backend=marlin` and the MTP `--speculative-config`) are
what make the MoE perform — they are recorded as catalog data
([`lobes/catalog.py`](../lobes/catalog.py)) and printed by
`lobes switch mmangkad/Qwen3.6-35B-A3B-NVFP4`, but are **not** in the default
single-model template (they break the dense/hybrid models, and compose can't
conditionally omit a flag). Add them to the compose `command` by hand:

```bash
vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --port 8000 --tensor-parallel-size 1 --trust-remote-code --dtype auto \
  --quantization modelopt --kv-cache-dtype fp8 \
  --attention-backend flashinfer --moe-backend marlin \
  --gpu-memory-utilization 0.85 --max-model-len 65536 \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
```

Output-token throughput across the three workloads (16 concurrent requests):

| workload | Blackwell 6000 Pro | DGX Spark | Jetson Thor |
|---|---|---|---|
| prompt-heavy (8K/1K) | 343.8 tok/s | 171.6 tok/s | 124.2 tok/s |
| decode-heavy (1K/8K) | 1052.7 tok/s | 268.2 tok/s | 239.1 tok/s |
| balanced (1K/1K) | 817.5 tok/s | 249.5 tok/s | 190.7 tok/s |

MTP speculative-decode acceptance was highest on the decode-heavy workload
(~80–84 %), lowest on balanced (~57–59 %). These are shahizat's numbers on
dedicated boxes (the `nvidia/` checkpoint, concurrency 16, **with** MTP) — see
[`tuning-profiles.md`](tuning-profiles.md) for how the `--purpose` knob maps to
these shapes.

## Live replication on this GB10 (2026-05-31)

We did not trust the posted numbers — we measured. On the shared DGX Spark
`spark-f8a9` (single GB10, 121.7 GiB unified, shared with the audio NIMs + reachy;
vLLM 0.19.0+nv26.04), with the 27B stopped and the recipe above **minus MTP** at
util 0.70 / 32768:

| Metric (single-stream, batch=1) | 35B MoE (no MTP) | 27B hybrid (primary) |
|---|---|---|
| decode throughput | **35.0 / 36.1 tok/s** | 7.8 / 7.9 tok/s |
| prefill (845 tok + 16 gen) | **0.62 s** | 2.33 s |

So the 35B MoE is **~4.6× faster on single-stream decode and ~3.8× faster on
prefill** than the 27B on the same box — the MoE's ~3B-active-params advantage,
reproduced. (`vllm bench serve` at concurrency 1 agrees: 34.7 tok/s, TTFT 0.70 s,
TPOT 28 ms.) We could **not** reproduce shahizat's exact figures — he ran the
`nvidia/` checkpoint on *dedicated* boxes at concurrency 16 **with** MTP (which
roughly doubles per-stream decode); our run is the `mmangkad/` copy on a *shared*
box, single-stream, **without** MTP (it does not load here). The qualitative
result — MoE = much faster decode — replicates; the headline tok/s does not, and
the gap is explained by box, concurrency, and the missing MTP draft.

## Why we serve the `mmangkad/` copy, not `nvidia/` (vLLM version, 2026-05-31)

shahizat used `nvidia/Qwen3.6-35B-A3B-NVFP4`. We tried to switch to it (and to a
newer vLLM) to get MTP working — and hit a hard wall on the GB10:

- **The `nvidia/` checkpoint will not load on the NGC image's vLLM 0.19.0.** Its
  NVFP4-MoE experts fail every backend: `marlin` / `flashinfer_trtllm` → "not
  supported for unquantized MoE"; `triton` / auto → `KeyError:
  layers.0.mlp.experts.w2_input_scale`. Both `--quantization modelopt` and
  `modelopt_fp4` behave the same.
- **A newer vLLM *does* run on the GB10.** A derived image with
  `pip install vllm==0.21.0` pulls upstream torch 2.11.0 + CUDA-13 wheels
  (aarch64 wheels exist); torch 2.11.0 works on the GB10 (`device_capability
  (12,1)`; `sm_121` is forward-compatible with its `sm_120` kernels — a GPU
  matmul ran). On 0.21.0 the quant is now **recognized** (`modelopt_mixed`), but
  the MoE expert loader still fails the same way (`marlin` → "unquantized";
  `triton`/auto → missing `w2_input_scale`).
- **0.22.0 / nightly are not pip-installable here** (aarch64): a
  `nvidia-cutlass-dsl[cu13]` dependency conflict with no matching distribution.

Net: the `nvidia/` checkpoint's MoE export needs a vLLM build with NVFP4-MoE
expert support that isn't installable on this Grace/Blackwell (aarch64) box yet.
shahizat's dedicated boxes were almost certainly x86, where a suitable vLLM
installs cleanly. **The working NVFP4 MoE on the GB10 remains the `mmangkad/`
copy** (loads on the stock NGC `26.04-py3` image with `--moe-backend marlin`,
~35 tok/s single-stream — above). Revisit `nvidia/` + MTP when a vLLM with the
right loader ships for aarch64 (a newer NGC image, or upstream ≥0.22 gaining
aarch64 wheels). The image stays **NGC `26.04-py3`** (latest tag; vLLM
0.19.0 + torch 2.12.0a0.nv26.04 + CUDA 13.2, all Blackwell-patched).

> **The 27B took the other route.** Rather than wait for a newer engine, the 27B
> gets MTP from a checkpoint that *ships the MTP draft weights*
> (`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`,
> [`qwen3.6-27b-text-nvfp4-mtp.md`](qwen3.6-27b-text-nvfp4-mtp.md), issue #26). The
> same strategy could unblock MTP here — a 35B re-export with the draft head
> grafted back, loadable on the stock `0.19.0` image — without the `nvidia/`
> NVFP4-MoE loader. Re-testing `nvidia/Qwen3.6-35B-A3B-NVFP4` + MTP is tracked as a
> follow-up.
