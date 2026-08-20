# Qwen3.6-35B-A3B-NVFP4: two checkpoints, two stories

The catalog carries **two, distinct** `Qwen3.6-35B-A3B-NVFP4` entries — same
architecture family (MoE, ~35B total / ~3B active per token), different
org/export, different role, different story:

- **`unsloth/Qwen3.6-35B-A3B-NVFP4`** — the **`worker`** role, the eighth
  first-class Colleague role (thor-worker-lobe plan). MULTIMODAL, ships its
  OWN self-hosted MTP draft, 262144 native context. See ["`worker`: the
  eighth Colleague role"](#worker-the-eighth-colleague-role-unslothqwen36-35b-a3b-nvfp4)
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
