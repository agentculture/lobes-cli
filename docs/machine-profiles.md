# Machine profiles — hardware detection and per-role tuning

lobes runs the **fleet** (four co-resident vLLM backends + a gateway, by
default — a mesh-brain deployment shape can drop one of them to a peer box;
see `docs/deployment-shapes.md`) with knob values tuned to the hardware it
lands on. A **machine profile** is the per-card tuning declaration: which
models serve each role (`cortex` / `senses` / `embedder` / `reranker`), their
GPU memory budget, context length, attention backend, and other vLLM knobs
the compose template substitutes. This document walks the detection flow,
how a profile is chosen, the knobs' meanings and provenance, and how to
write custom profiles for new hardware.

See `lobes explain profiles` or `lobes explain tuning` for the brief version.
That command reads from `lobes/explain/catalog.py`; this file is the deep
reference.

## How detection works

The card-detection flow below runs only during **`lobes init`** — either
auto-detected from the card, or forced with an explicit `--profile <name>` —
to resolve the per-role `Profile` this document describes and render it into
`.env`. **`lobes serve` and `lobes status` run no detection at all** and
accept neither `--profile` nor `--machine`: they only read the deployment's
already-rendered `.env` (including the persisted `LOBES_PROFILE`) — profile
selection happens once, at `init` time, not on every `serve`/`status` call.
`lobes switch` has its own, separate `--machine` flag — a *different*,
legacy single-model machine-profile knob (`lobes/profiles/__init__.py`'s
`MachineProfile` / `resolve_machine()`, tuning the single-model template's
GPU/context defaults), not the fleet `Profile` system covered here; left at
its default (`auto`), it re-detects the GPU name via `nvidia-smi -L` on its
own code path, not `lobes/runtime/_detect.py::detect_card()`.

The following steps resolve the machine name at `lobes init` time:

1. **Gather raw facts** (from `lobes/runtime/_detect.py::detect_card()`):
   - **Device name**: from `nvidia-smi --query-gpu=name,compute_cap`
   - **Compute capability**: from the same nvidia-smi query (e.g., `"11.0"` →
     `"sm_110"`)
   - **Total memory**: from `/proc/meminfo MemTotal` (honest system total on
     unified-memory Jetson boards; nvidia-smi memory fields report `[N/A]` on
     these boards, so we never parse them)
   - **Hostname**: from `socket.gethostname()` (e.g., `"thor"`)
   - **Device-tree model** (Jetson boards only): from `/proc/device-tree/model`,
     NUL-terminated (e.g., `"NVIDIA Jetson AGX Thor"`)

   Every probe is independent and degrades gracefully: a missing binary, a
   timeout, a missing file, or unparsable output never raises. The
   corresponding fact is simply `None`, and resolution continues.

2. **Resolve to a card name** (from `lobes/machines/_registry.py::detect()`):
   - Pass the gathered facts to the live `lobes.machines` registry (a dict of
     `CardStrategy` instances, one per supported chip, in detection-precedence
     order).
   - The first card's `DetectionSignature` that matches wins. Signatures are
     matched by `name_markers` (device name or hostname substrings) or
     `compute_capability` (e.g., `"sm_110"`).
   - If no card matches, the result is `UNKNOWN`.

3. **Never guess a fallback silently.** An `UNKNOWN` result is **first-class**
   and honest — it is not replaced by a "closest" card. A deployment on
   unrecognized hardware gets an explicit warning (in `lobes init` / `lobes
   status` output), and the conservative `base` profile is used to avoid OOM
   crashes on first boot.

## How a profile is chosen

Once detection (or an explicit `--profile` flag on `lobes init`) resolves the
card name, a profile is looked up via `lobes/profiles/loader.py::resolve_profile()`:

1. **Explicit always wins**: if `--profile <name>` was given (or a
   `LOBES_PROFILE` env var), that profile is used, even if it diverges from
   the auto-detected machine name. A warning is printed if forced.

2. **Operator override next**: if a file
   `<deployment-dir>/profiles/<name>.toml` exists (where `<name>` is the
   machine name or explicit `--profile`), it wins over the built-in.

3. **Built-in fallback**: the packaged profile in `lobes/profiles/builtin/`,
   resolved by name.

A same-named operator file **completely overrides** the built-in — they are
never merged field-by-field. The operator profile is a self-contained
declaration; any role or knob it stays silent on falls back to "no opinion",
meaning the compose template's own `${VAR:-default}` applies.

## The knob reference

Every knob is optional (`None` = "profile takes no position, template default
applies"). Only knobs the profile diverges on appear in the TOML or env; the
rest are inherited from the template.

The four roles and seven knobs map to env vars via `lobes/profiles/render.py`:

| role | env prefix | feasible | model | gpu_mem_util | max_model_len | quantization | kv_cache_dtype | attention_backend | enforce_eager | max_num_seqs |
|---|---|---|---|---|---|---|---|---|---|---|
| `cortex` | `PRIMARY_` | `FEASIBLE` | `MODEL` | `GPU_MEM_UTIL` | `MAX_MODEL_LEN` | `QUANTIZATION` | `KV_CACHE_DTYPE` | `ATTENTION_BACKEND` | `ENFORCE_EAGER` | `MAX_NUM_SEQS` |
| `senses` | `MULTIMODAL_` | `FEASIBLE` | `MODEL` | `GPU_MEM_UTIL` | `MAX_MODEL_LEN` | `QUANTIZATION` | `KV_CACHE_DTYPE` | `ATTENTION_BACKEND` | (not used) | `MAX_NUM_SEQS` |
| `embedder` | `EMBED_` | `FEASIBLE` | `MODEL` | `GPU_MEM_UTIL` | `MAX_MODEL_LEN` | (not used) | (not used) | `ATTENTION_BACKEND` | (not used) | (not used) |
| `reranker` | `RERANK_` | `FEASIBLE` | `MODEL` | `GPU_MEM_UTIL` | `MAX_MODEL_LEN` | (not used) | (not used) | `ATTENTION_BACKEND` | `ENFORCE_EAGER` | (not used) |

### Field descriptions and provenance

#### `feasible` (default `True`)

**What it does:** If `false`, the role is marked infeasible on this box.
`lobes/profiles/render.py` renders it as `<PREFIX>_FEASIBLE=false` in the
`.env`, and (once wired up in t6) the gateway omits it from `GET /capabilities`
and returns `404 role_infeasible` on POST requests. If `true` (the default),
no `<PREFIX>_FEASIBLE` key is emitted — "feasible" is the assumed default.

**When set:**

- `spark` profile (GB10): all four roles `feasible=true` (all roles load-tested
  here).
- `thor` profile (Jetson AGX Thor): all four roles `feasible=true`
  (validated live 2026-07-13).
- `base` profile (unknown card): `cortex=true`, `senses=false`, `embedder=true`,
  `reranker=true` — the multimodal gear is disabled to save memory on unknown
  hardware.

#### `model` (no default)

**What it does:** The HuggingFace model id (e.g.,
`sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP`). Rendered to *two* env vars:
`<PREFIX>_MODEL` and `<PREFIX>_SERVED_NAME` (both set to the same value). The
compose template passes the served name to vLLM's `--served-model-name`
separately from the model id it downloads; the two must agree for the gateway
to route correctly.

**When set:**

- `spark` cortex: `sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP` (the 256K-native
  MTP primary, re-exported with its draft head restored; load-tested 2026-05-31
  on GB10, ~2.4x single-stream decode over the archived 27B baseline).
- `spark` senses: `coolthor/gemma-4-12B-it-NVFP4A16` (12B multimodal, load-tested
  on GB10).
- `spark` embedder: `Qwen/Qwen3-Embedding-0.6B` (0.6B dense embedder, 1024-dim,
  32K context).
- `spark` reranker: `Qwen/Qwen3-Reranker-0.6B` (0.6B cross-encoder).
- `thor` cortex/senses/embedder/reranker: same as spark (both are 128 GB unified
  Blackwell-class boards; Thor diverges only on vLLM knobs, not model ids).
- `base` cortex: `Qwen/Qwen3.5-4B` (small model for unknown hardware — see
  `docs/qwen3.5-4b-minor.md`, 256K native, bf16).
- `base` senses: omitted (infeasible; see `feasible=false` above).
- `base` embedder/reranker: same as spark.

#### `gpu_mem_util` (range 0.0–1.0, typically 0.06–0.85)

**What it does:** The fraction of total GPU memory lobes tells vLLM the
generate/pooling lane may use. vLLM uses this to cap `--max-num-seqs` (batch
size) and prefill throughput so OOM never happens. On unified-memory boards
(Spark, Thor) the "GPU memory" is actually a carved-out slice of the unified
pool; on dedicated-VRAM boards (RTX PRO 6000) it is true discrete VRAM.

**When set and why:**

- `spark` cortex: `0.30` — the primary serves at this utilization budget
  (`0.30 + 0.14 + 0.06 + 0.06 = 0.56` total across all four roles on 128 GB),
  chosen to leave headroom for other mesh services on the shared GB10
  (load-tested 2026-05-31).
- `spark` senses: `0.14` — the 12B multimodal gear, smaller budget than cortex.
- `spark` embedder/reranker: `0.06` each — the two ~0.6B pooling gears.
- `thor` cortex/senses/embedder/reranker: same as spark (Thor is also 128 GB
  unified; the budget is hardware-independent, not a divergence).
- `base` cortex: `0.30` — safe default for unknown hardware (small model, so
  conservative util).
- `base` embedder/reranker: `0.06` each.

#### `max_model_len` (typical values: 8192–262144 tokens)

**What it does:** The maximum sequence length (in tokens) vLLM will accept for
this role. Determines GPU memory overhead; longer sequences = larger KV cache.
Tuned per card to balance headroom and capability.

**When set and why:**

- `spark` cortex: `131072` (128K) — the primary's load-tested context on the
  GB10 (the checkpoint has 256K native, but at util 0.30 on a shared board,
  128K is the validated cap; issue #107 tracks broader context migration).
- `spark` senses: `32768` (32K) — the multimodal gear's context.
- `spark` embedder/reranker: `8192` (8K) — the pooling gears' context.
- `thor` cortex: `131072` (128K) — same as spark (both boards are 128 GB, same
  fleet budget; cortex is util-bound, not context-bound at these numbers).
- `thor` senses/embedder/reranker: same as spark.
- `base` cortex: `32768` (32K) — the small 4B model's cap on unknown hardware,
  conservative.
- `base` embedder/reranker: `8192` (8K) — same as spark.

#### `quantization` (typical values: `"modelopt"`, `"compressed-tensors"`, or omitted)

**What it does:** The quantization format. Passed to vLLM's `--quantization`
flag. Per-model semantics (defined in `lobes/catalog.py`); some models have
multiple quantization options.

**When set and why:**

- `spark` cortex: `"modelopt"` — the MTP primary is an nvidia/ ModelOpt FP4
  export; requires this quantization.
- `spark` senses: `"compressed-tensors"` — the Gemma 4 checkpoint uses
  compressed-tensors format.
- `spark` embedder/reranker: omitted (these models do not require explicit
  quantization on vLLM nv26.04).
- `thor` cortex/senses: same as spark.
- `thor` embedder/reranker: same as spark.
- `base` cortex: omitted (the 4B model is bf16/unquantized, not quantized; see
  `docs/qwen3.5-4b-minor.md`).
- `base` embedder/reranker: omitted (same as spark).

#### `kv_cache_dtype` (typical values: `"fp8"`, `"auto"`, or omitted)

**What it does:** The dtype of the KV cache buffers in vLLM. Traded against
accuracy for memory savings. Passed to vLLM's `--kv-cache-dtype` flag.

**When set and why:**

- `spark` cortex: `"fp8"` — FP8 KV cache saves memory and is validated on this
  checkpoint/board pairing (load-tested 2026-05-31).
- `spark` senses: omitted (the compose template's default, typically `"float16"`
  or None, applies).
- `spark` embedder/reranker: omitted.
- **`thor` cortex: `"auto"`** — the hand-validated value the live Thor fleet
  runs with. The checkpoint ships no calibrated KV scales
  (`kv_cache_quant_algo: null`); an earlier session recorded an
  `assert layer.k_scale > 0.0` crash under `fp8` on Thor, but on the currently
  pinned nightly that assert did **not** reproduce (2026-07-13): `fp8` now
  boots with "Using KV cache scaling factor 1.0" / "uncalibrated q_scale"
  warnings — an **accuracy risk**, not a crash. `auto` serves the KV cache in
  the model dtype and avoids the uncalibrated-fp8 question entirely.
  **Provenance:** the hand-edited deployment that ran this box for weeks, plus
  the 2026-07-13 boot logs recorded in issue #109 (which also tracks the GB10
  side of the question).
- `thor` senses/embedder/reranker: omitted (same as spark).
- `base` cortex/senses/embedder/reranker: omitted.

#### `attention_backend` (typical values: `"flashinfer"`, `"TRITON_ATTN"`)

**What it does:** The attention kernel backend. vLLM chooses at inference time
based on the attention shape and the backend's capabilities; the flag provides
an override or force. Affects both speed and stability.

**When set and why:**

- `spark` cortex: omitted (vLLM auto-selects; FlashInfer on the GB10).
- `spark` **and** `thor` senses: `"TRITON_ATTN"` — **not a Thor divergence.**
  This mirrors the fleet template's cross-machine default for the Gemma gear:
  Gemma 4's heterogeneous head sizes (sliding 256 / full-attention 512) need
  the Triton backend. Same value on every machine; its env plumbing
  (`MULTIMODAL_ATTENTION_BACKEND`) is unchanged pending the GB10 check in
  issue #109.
- `spark` embedder/reranker: omitted (vLLM auto-selects on the GB10).
- **`thor` cortex: omitted** (the generate lane runs stably with the
  auto-selected backend on Thor).
- **`thor` embedder: `"TRITON_ATTN"`** — a real sm_110 divergence: the
  auto-picked FLASH_ATTN pooling path **hangs the forward pass** on sm_110
  (requests accepted, never answered — `/health` stays green, which is why the
  correctness probes exist). Inherited from the shared `SM_110` trait in
  `lobes/machines/_traits.py`, overlaid by `lobes/profiles/loader.py`.
- **`thor` reranker: `"TRITON_ATTN"`** — the same sm_110 pooling bug, surfaced
  as NaN relevance scores (wrong orderings, issue #105) rather than a hang.
  Inherited from the SM_110 trait.
- `base` cortex/senses/embedder/reranker: omitted (template defaults apply; the
  conservative small model on unknown hardware avoids tuning for any backend).

#### `enforce_eager` (boolean: `true` or `false`; renders to `"--enforce-eager"`/`"--no-enforce-eager"`)

**What it does:** If `true`, vLLM disables CUDA graph capture and runs every
request in eager mode. Slower but more stable. Passed to vLLM as
`--enforce-eager` (or `--no-enforce-eager` if `false`). Note: the env var
holds the **token**, not a bare boolean (see `lobes/profiles/render.py` for
the translation).

**When set and why:**

- `spark` cortex/senses/embedder: omitted (CUDA graphs are stable on GB10).
- **`spark` reranker: omitted** (template default; CUDA graphs work here).
- **`thor` reranker: `true`** — validated live on Thor 2026-07-13. With Triton
  attention, the reranker's CUDA-graph capture is **unstable on sm_110**:
  requests fail with `cudaErrorLaunchFailure` when CUDA graphs are enabled.
  Enforce eager mode (no graphs) and the reranker runs stably and correctly.
  **Provenance:** live reranker correctness probe (ordering test) crashed with
  graph capture enabled but passed with `enforce_eager=true`; see issue #105
  comment for the crash log. Inherited from the SM_110 trait.
- `thor` cortex/senses/embedder: omitted (CUDA graphs are stable elsewhere on
  Thor).
- `base` cortex/senses/embedder/reranker: omitted.

#### `max_num_seqs` (typical values: 2, 4, 8)

**What it does:** The maximum number of sequences (batch size) vLLM will
schedule concurrently in a single forward pass. Tuned per workload
(`--purpose`), not per machine. Affects both throughput (more batches) and
latency (higher tail latency per request).

**When set and why:**

- `spark` cortex: `2` — the fleet's cortex defaults to small batches for low
  latency (appropriate for the agentic workload — agents mostly read one
  response at a time, not many concurrent requests).
- `spark` senses/embedder/reranker: omitted (template defaults apply, typically
  4–8 for pooling layers).
- `thor` cortex: `2` — same as spark (machine-independent; this knob is
  workload-tuned, not machine-tuned, so thor and spark match).
- `thor` senses/embedder/reranker: omitted (same as spark).
- `base` cortex/senses/embedder/reranker: omitted.

## Writing your own profile

Operator-defined profiles go in `<deployment-dir>/profiles/<name>.toml` (the
file's stem is the profile name). They are discovered automatically by
`lobes init` (the only verb that resolves a `Profile` — see "How detection
works" above) and override any built-in with the same name.

A profile is a self-contained TOML declaration. Example:

```toml
name = "my-custom-box"
summary = "Custom RTX 6000 tuning with larger context"

[roles.cortex]
feasible = true
model = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
gpu_mem_util = 0.75
max_model_len = 180000
quantization = "modelopt"
kv_cache_dtype = "fp8"
attention_backend = "flashinfer"

[roles.senses]
feasible = true
model = "coolthor/gemma-4-12B-it-NVFP4A16"
gpu_mem_util = 0.15
max_model_len = 32768
# attention_backend inherited from template default

[roles.embedder]
feasible = true
model = "Qwen/Qwen3-Embedding-0.6B"
gpu_mem_util = 0.05
max_model_len = 8192

[roles.reranker]
feasible = true
model = "Qwen/Qwen3-Reranker-0.6B"
gpu_mem_util = 0.05
max_model_len = 8192
```

**Rules:**

- **File name is the profile name.** `profiles/my-custom-box.toml` → profile
  name is `my-custom-box`. Use it as `lobes init --profile my-custom-box --apply`
  (the flag is `--profile` on `init`, `--machine` on `switch`).
- **Inline `name` field must match the file stem.** If the file declares
  `name = "something-else"`, profile loading fails with an error (this prevents
  confusion from stale copies).
- **Roles are optional.** A role omitted from the TOML means "take no position
  — use the template's defaults for this role." You don't have to restate all
  four roles; a minimal profile might only touch `cortex`.
- **Knobs within a role are optional.** A knob omitted means "no opinion" (the
  template's `${VAR:-default}` applies). This lets you override only the knobs
  you care about and inherit sensible defaults for the rest.
- **Unknown knob names are errors.** A typo in a knob name (e.g.,
  `gpu_memory_util` instead of `gpu_mem_util`) causes profile loading to fail
  with remediation (list of known knobs). This is strict-by-design to catch
  operator mistakes early.
- **Profile override is complete, not merged.** If you define a custom
  `my-custom-box` profile and a built-in `my-custom-box` exists, the custom
  one replaces it entirely; the two are never merged field-by-field.

To use your profile:

```bash
# Explicit flag (--profile on init, --machine on switch) or LOBES_PROFILE env var:
lobes init --profile my-custom-box --apply
lobes switch my-model --machine my-custom-box --apply

# Automatic if the detected machine name matches:
# (detection resolves "my-custom-box" → looks for profile "my-custom-box" →
# finds it in <deploy-dir>/profiles/my-custom-box.toml)
```

Auto-detection only works if the *detected machine name* (from
`lobes/machines/_registry.py`) matches the profile name. If you want a custom
profile to be auto-selected on a new card, add a new `CardStrategy` module to
`lobes/machines/` (following the `spark.py` / `thor.py` pattern).

For a complete worked example — a live-validated operator profile for the
Jetson AGX Orin 64GB (Gemma `senses` at its full 128K context on Ampere sm_87,
with measured knob values and the Jetson divergences found on the way) — see
[`orin-profiles.md`](orin-profiles.md).

## The goldens contract

CI guards against cross-machine breakage via golden `.env` files — one per
built-in profile. Every PR that edits profile or machine-strategy code must
regenerate them:

```bash
uv run python tests/goldens/regen.py
```

This reads every built-in profile, renders it to `.env` format, and byte-diffs
against `tests/goldens/<profile>.env` (e.g., `tests/goldens/spark.env`).

**The invariant:** editing the `thor` profile or the SM_110 trait must not
change the spark golden (i.e., `spark.env` stays identical). This ensures that
work on one machine's profile does not accidentally break another's. If you edit
machine strategy or profile code and the golden diff fails CI, re-run the regen
script (it updates the goldens to the new render output) and commit the change.

The `template-defaults.env` golden verifies that the compose template's own
built-in defaults (when no profile is loaded) render as expected.

## Support table

lobes is deployed and validated on the following hardware, with the stated
caveat profile as the truthful, tested configuration. Aspirational entries are
marked clearly as **unvalidated**. **This table's scope is BUILT-IN
profiles/shapes** (`lobes/profiles/builtin/*.toml`,
`lobes/profiles/builtin_shapes/*.toml`) — the #108 rule is that a built-in
earns "validated" status only via a repo-recorded boot of that *exact*
built-in, so a validated operator-defined profile (`<deploy-dir>/profiles/
<name>.toml`) does not by itself validate a built-in for the same card. The
Orin row below carries both facts side by side so they read as complementary,
not contradictory.

| card | machine | profile | status | validation |
|---|---|---|---|---|
| **DGX Spark** (Grace Blackwell, 128 GB unified) | `spark` | `spark.toml` | load-tested | 2026-06-03 — `lobes benchmark` ran at ~7.8–8.0 tok/s decode (27B primary, util 0.30, single-stream) on the fleet duo (cortex 128K, senses 32K) with FlashInfer attention. **The correctness probes postdate that run and have not been executed on the GB10** — rerank ordering there is explicitly unverified (issue #106). See `docs/tuning-profiles.md` and the Spark run on `docs/qwen3.6-27b-text-nvfp4-mtp.md`. |
| **Jetson AGX Thor** (Blackwell-class sm_110, 128 GB unified) | `thor` | `thor.toml` | load-tested | 2026-07-13 — the three correctness probes (cortex known-answer, embed paraphrase-ranking, rerank ordering) all pass with the profile's four divergences (`cortex kv_cache_dtype=auto`, `embedder TRITON_ATTN`, `reranker TRITON_ATTN + enforce_eager=true`); senses' health was not confirmed in that run (it boots last and the concurrent-boot race, caveat 4 below, can catch it). |
| **unknown card** (UNKNOWN) | `generic` | `base.toml` | conservative fallback | — — no hardware beyond Spark/Thor is load-tested; UNKNOWN cards are served a small 4B model, no 27B, and no multimodal (`senses` disabled). Avoids OOM crashes on first boot. See issue #107 for broader "small default on every card" work. |
| Jetson AGX Orin / Orin Nano Super | `(not yet)` | (not yet) | **no built-in profile — unvalidated at that scope; operator-validated separately** | **Built-ins:** no built-in `orin` profile exists, and the built-in `orin-small` *shape* remains DECLARED/UNVALIDATED (#108 — an unbooted built-in stays unvalidated regardless of operator activity elsewhere). Do not claim built-in Orin support from this row. **Operator deployment:** a physical Jetson AGX Orin 64GB WAS operator-validated 2026-07-16/17, using a hand-written **operator profile** (`<deploy-dir>/profiles/orin.toml` — `senses`/`embedder`/`reranker`, `senses` at its full native 131072 context, measured `gpu_mem_util=0.45`) composed with the built-in `thor-lobe` *shape* (not `orin-small`). See [`orin-profiles.md`](orin-profiles.md) for the measured knobs and the Jetson/sm_87 divergences found live. |

## Thor caveats and validation facts

This section documents Thor's validated divergences from Spark's baseline,
with honest provenance. Thor is load-tested and ready for production, but with
these specific knob values; they exist because of real hardware constraints.

### 1. `cortex kv_cache_dtype=auto` (not `fp8`)

**The problem:** The checkpoint ships no calibrated KV scales
(`kv_cache_quant_algo: null`). An earlier session on this box recorded an
`assert layer.k_scale > 0.0` crash under the Spark default `fp8`; on the
currently pinned nightly that assert did **not** reproduce (2026-07-13) —
`fp8` boots with `Using KV cache scaling factor 1.0 for fp8_e4m3` /
`uncalibrated q_scale` warnings instead, i.e. an **accuracy risk** rather
than a crash. Whether the GB10 has the same exposure is the open half of
issue #109.

**The fix:** Set `cortex kv_cache_dtype=auto` — the KV cache is kept in the
model dtype, sidestepping uncalibrated-fp8 entirely.

**Validation:** the hand-tuned deployment that has served this box runs
`auto`; on the clean-boot validation (2026-07-13) the cortex known-answer
probe passed under `auto`. No probe was run under `fp8` — the `fp8` evidence
is boot-log warnings, not a measured accuracy delta.

**Provenance:** Measured on Thor (sm_110, 128 GB unified, Jetson AGX).
Carried by `lobes/machines/thor.py` and overlaid by
`lobes/profiles/loader.py` from the live registry (not hardcoded in TOML).

### 2. `embedder` / `reranker` use `TRITON_ATTN` (an sm_110 divergence)

**The problem:** On sm_110 (Thor), the auto-picked FLASH_ATTN pooling path is
broken: on the **embedder** it hangs the forward pass (requests accepted,
never answered — `/health` stays green, which is exactly why the correctness
probes exist); on the **reranker** the same bug surfaces as NaN relevance
scores, i.e. wrong orderings (issue #105).

**The fix:** Force `attention_backend=TRITON_ATTN` for the two pooling roles.
(`senses` also runs `TRITON_ATTN`, but that is the cross-machine template
default for Gemma 4's heterogeneous head sizes — identical on Spark, not a
Thor divergence.)

**Validation:** Live correctness probes on the clean-boot validation
(2026-07-13): the embed paraphrase probe passed (cos(paraphrase) 0.89 >
cos(unrelated) 0.23, no hang) and the rerank ordering probe ranked the
relevant document first, with no `cudaErrorLaunchFailure` (see caveat 3).

**Provenance:** The SM_110 trait (a compute-capability characteristic, not
Thor-only — a future sm_110 board inherits it) in
`lobes/machines/_traits.py` defines the knobs for both pooling roles;
`lobes/machines/thor.py` composes the trait. Issue #105 documents the
symptoms; issue #106 tracks checking rerank ordering on the GB10 (still
open).

### 3. `reranker enforce_eager=true` (CUDA graph capture disabled)

**The problem:** On sm_110, even with Triton attention, the reranker's CUDA
graph capture is unstable. When graph capture is enabled (the default), some
requests fail with `cudaErrorLaunchFailure`.

**The fix:** Set `reranker enforce_eager=true`. This disables CUDA graph
capture and runs every request in eager mode, avoiding the error.

**Validation:** The `cudaErrorLaunchFailure` under CUDA graphs was recorded in
issue #105 (earlier session on this box). On the clean-boot validation
(2026-07-13) with `enforce_eager=true` the rerank ordering probe passed with
no engine crash.

**Provenance:** Measured on Thor. The SM_110 trait in `lobes/machines/_traits.py`
defines `reranker enforce_eager=true` as a sm_110 characteristic. It is
inherited by the Thor `CardStrategy` (composes the trait).

### 4. Concurrent first boot can fail on a memory race (any profile)

**The problem (measured 2026-07-13, 4/4 concurrent-boot attempts failed):**
when all gears start at once, each vLLM engine's memory-profiling window sees
the *other* gears' weight loads — on Jetson unified memory, weight files read
through the page cache count as occupied — so the cortex fails either the
free-memory gate (`Free memory on device … is less than desired GPU memory
utilization`) or KV allocation (`No available memory for the cache blocks`),
**regardless of KV dtype or profile**. The same values boot cleanly when the
primary starts alone: this is a bring-up race, not a steady-state
misconfiguration. Bring-up ordering is not expressible as a per-gear env knob
— tracked as follow-up work (plan risk r7).

**Workaround (validated):**

```sh
# after any teardown, reclaim page cache before rebooting the fleet:
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
# bring the primary up first, then the rest:
docker compose up -d vllm-primary   # wait until healthy
docker compose up -d
```

With that sequence the thor profile boots clean and all three correctness
probes pass. `senses` (the Gemma 12B) boots **last** and is the gear most
often caught by the race — its health was not confirmed in the 2026-07-13
clean-boot run; the steady-state values (32K / util 0.14) are the ones the
long-running hand-tuned deployment serves with.

### Note on future verification

- **Issue #106** tracks re-verifying the Triton-ATTN knobs on GB10 (Spark) to
  confirm they do not cause regressions there. They are expected to work (Spark
  can run FlashInfer fine, so Triton is not needed, but forcing it anyway
  should not break correctness).
- **Issue #109** tracks two GB10 verifications: whether uncalibrated-fp8 KV is
  an accuracy risk there too (the Thor crash originally attributed to it did
  not reproduce on the pinned nightly — fp8 now warns instead of asserting),
  and whether the legacy `VLLM_ATTENTION_BACKEND` env is truly dead on the
  GB10's pinned image before the template drops it.
- **Issue #107** tracks broader work to tuned-small-model defaults for every
  card, not just UNKNOWN. Spark and Thor are mature; this is future work.

## See also

- `lobes explain tuning` / `lobes explain profiles` — brief in-CLI reference
- `lobes explain machine` — machine profile flags and detection
- `docs/tuning-profiles.md` — workload (`purpose`) profiles and shahizat's
  benchmark baseline
- `docs/colleague-stack.md` — the fleet contract (roles, context, memory budget)
- `docs/deployment-shapes.md` — the orthogonal deployment-shape axis (which
  roles a box hosts at all, composed over this per-machine tuning)
- `lobes/machines/` — CardStrategy modules (one per chip; the source of truth
  for detection signatures and machine knobs)
- `lobes/profiles/` — profile schema, loader, renderer, and built-in TOML
- `lobes/runtime/_detect.py` — the detection fact-gatherer
- Issue #105 — reranker `cudaErrorLaunchFailure` on Thor (fixed by
  enforce_eager + TRITON_ATTN)
- Issue #106 — GB10 re-verification of TRITON_ATTN knobs (pending)
- Issue #107 — broader tuned-small-model defaults for every card (future work)
- Issue #109 — GB10 verifications: uncalibrated-fp8 KV accuracy exposure, and
  whether `VLLM_ATTENTION_BACKEND` is dead on the pinned image
