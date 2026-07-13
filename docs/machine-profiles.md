# Machine profiles — hardware detection and per-role tuning

lobes runs the **fleet** (four co-resident vLLM backends + a gateway) with knob
values tuned to the hardware it lands on. A **machine profile** is the per-card
tuning declaration: which models serve each role (`cortex` / `senses` /
`embedder` / `reranker`), their GPU memory budget, context length, attention
backend, and other vLLM knobs the compose template substitutes. This document
walks the detection flow, how a profile is chosen, the knobs' meanings and
provenance, and how to write custom profiles for new hardware.

See `lobes explain profiles` or `lobes explain tuning` for the brief version.
That command reads from `lobes/explain/catalog.py`; this file is the deep
reference.

## How detection works

When `lobes init` runs (or `lobes serve` / `lobes status` without an explicit
`--machine`), the following steps resolve the machine name:

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

Once detection (or an explicit `--machine` flag) resolves the card name, a
profile is looked up via `lobes/profiles/loader.py::resolve_profile()`:

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
- **`thor` cortex: `"auto"`** — validated live on Thor 2026-07-13. The fp8
  default asserts `k_scale > 0.0` on this checkpoint/board pairing, and that
  assertion fails with uncalibrated fp8 on Thor (issue #109). The `"auto"` mode
  falls back to float16 when the assertion would fail, avoiding the crash.
  **Provenance:** live correctness probe (cortex known-answer test) passed with
  `kv_cache_dtype=auto` but failed with `fp8`; see issue #109 comment for the
  scale warning.
- `thor` senses/embedder/reranker: omitted (same as spark).
- `base` cortex/senses/embedder/reranker: omitted.

#### `attention_backend` (typical values: `"flashinfer"`, `"TRITON_ATTN"`)

**What it does:** The attention kernel backend. vLLM chooses at inference time
based on the attention shape and the backend's capabilities; the flag provides
an override or force. Affects both speed and stability.

**When set and why:**
- `spark` cortex: omitted (the template default `"flashinfer"` is used; the GB10
  is Blackwell-class and runs FlashInfer stably).
- `spark` senses: `"TRITON_ATTN"` — the multimodal/pooling lanes use Triton
  instead of FlashInfer. This was validated for the fleet's pooling path and
  keeps the attention layer consistent across roles.
- `spark` embedder/reranker: omitted (template default applies, but in practice
  vLLM's scheduler may choose Triton for pooling-shaped requests anyway).
- **`thor` cortex: omitted** (the template default applies; generate-lane
  attention works stably on Thor).
- **`thor` senses: `"TRITON_ATTN"`** — validated live on Thor 2026-07-13. The
  FlashInfer/FLASH_ATTN pooling path **hangs on sm_110** (Jetson Thor's compute
  capability). Force Triton instead. **Provenance:** live test observed hang
  with FlashInfer; switch to Triton fixed it (no hang, correct embeddings);
  explicitly declared in `lobes/profiles/builtin/thor.toml` (line 35) and
  referenced in issue #105.
- **`thor` embedder: `"TRITON_ATTN"`** — same reason as senses (SM_110 pooling
  issue). Inherited from the shared `SM_110` trait in
  `lobes/machines/_traits.py` and overlaid by `lobes/profiles/loader.py`.
- **`thor` reranker: `"TRITON_ATTN"`** — same reason. Inherited from SM_110
  trait and overlaid by loader.
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
`lobes init` / `lobes serve` and override any built-in with the same name.

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
  name is `my-custom-box`. Use it as `lobes init --machine my-custom-box --apply`.
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
# Explicit --machine or LOBES_PROFILE env var:
lobes init --machine my-custom-box --apply
lobes switch my-model --machine my-custom-box --apply

# Automatic if the detected machine name matches:
# (detection resolves "my-custom-box" → looks for profile "my-custom-box" →
# finds it in <deploy-dir>/profiles/my-custom-box.toml)
```

Auto-detection only works if the *detected machine name* (from
`lobes/machines/_registry.py`) matches the profile name. If you want a custom
profile to be auto-selected on a new card, add a new `CardStrategy` module to
`lobes/machines/` (following the `spark.py` / `thor.py` pattern).

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
marked clearly as **unvalidated**.

| card | machine | profile | status | validation |
|---|---|---|---|---|
| **DGX Spark** (Grace Blackwell, 128 GB unified) | `spark` | `spark.toml` | load-tested | 2026-06-03 — `lobes benchmark` ran at ~7.8–8.0 tok/s decode (27B primary, util 0.30, single-stream) on the fleet duo (cortex 128K, senses 32K) with FlashInfer attention. Correctness probes (cortex known-answer, embed paraphrase-ranking, rerank ordering) all pass. See `docs/tuning-profiles.md` and the Spark run on `docs/qwen3.6-27b-text-nvfp4-mtp.md`. |
| **Jetson AGX Thor** (Blackwell-class sm_110, 128 GB unified) | `thor` | `thor.toml` | load-tested | 2026-07-13 — all three correctness probes pass with the profile's four divergences (`cortex kv_cache_dtype=auto`, `senses/embedder/reranker TRITON_ATTN`, `reranker enforce_eager=true`). See the Thor caveats section below for the measured issues these divergences fix. |
| **unknown card** (UNKNOWN) | `generic` | `base.toml` | conservative fallback | — — no hardware beyond Spark/Thor is load-tested; UNKNOWN cards are served a small 4B model, no 27B, and no multimodal (`senses` disabled). Avoids OOM crashes on first boot. See issue #107 for broader "small default on every card" work. |
| Jetson AGX Orin / Orin Nano Super | `(not yet)` | (not yet) | **unvalidated** | Not deployed or tested. Named in the mesh topology for future integration but no profile yet. Do not claim Orin support. |

## Thor caveats and validation facts

This section documents Thor's validated divergences from Spark's baseline,
with honest provenance. Thor is load-tested and ready for production, but with
these specific knob values; they exist because of real hardware constraints.

### 1. `cortex kv_cache_dtype=auto` (not `fp8`)

**The problem:** The Spark profile's default `kv_cache_dtype=fp8` works on
GB10 but crashes on Thor with an uncalibrated-checkpoint scale assertion at
startup (see issue #109).

**The fix:** Set `cortex kv_cache_dtype=auto`. In `auto` mode, vLLM checks the
scale at startup and falls back to float16 if the assertion would fail,
avoiding the crash.

**Validation:** Live correctness probe (cortex known-answer test: `17 * 23 =
391`) passed with `kv_cache_dtype=auto` on Thor 2026-07-13 (see issue #109
comment for the exact log). The `auto` fallback is conservative (slower than
fp8) but correct.

**Provenance:** Measured on Thor (sm_110, 128 GB unified, Jetson AGX).
Referenced in `lobes/machines/thor.py:role_overrides` and overlaid by
`lobes/profiles/loader.py` from the live registry (not hardcoded in TOML).

### 2. `senses` / `embedder` / `reranker` use `TRITON_ATTN` (not FlashInfer)

**The problem:** On sm_110 (Thor), the FlashInfer/FLASH_ATTN attention backend
**hangs indefinitely** when the request shape matches the pooling pattern
(small batch, many tokens — e.g., embedding many documents). This affects the
`senses` (multimodal), `embedder`, and `reranker` roles.

**The fix:** Force `attention_backend=TRITON_ATTN` for all three roles.

**Validation:** Live correctness probes passed:
- `senses` (embed paraphrase-ranking test): compared two documents, ranked
  them, and verified the rank was correct. No hang, correct embeddings.
- `reranker` (ordering test): re-ranked items and verified the order. No hang,
  correct ranking (see caveat 3, below).
- Both ran without `cudaErrorLaunchFailure`.

**Provenance:** The SM_110 trait (shared hardware characteristic, not Thor-only)
in `lobes/machines/_traits.py` defines the knobs for both pooling roles.
Referenced in `lobes/machines/thor.py` (composes the trait). Issue #105
documents the hang symptom; issue #106 tracks GB10 re-verification (still open
pending Spark testing).

### 3. `reranker enforce_eager=true` (CUDA graph capture disabled)

**The problem:** On sm_110, even with Triton attention, the reranker's CUDA
graph capture is unstable. When graph capture is enabled (the default), some
requests fail with `cudaErrorLaunchFailure`.

**The fix:** Set `reranker enforce_eager=true`. This disables CUDA graph
capture and runs every request in eager mode, avoiding the error.

**Validation:** Live correctness probe (reranker ordering test) crashed with
`cudaErrorLaunchFailure` when graphs were enabled, but passed all ranking
assertions with `enforce_eager=true` on Thor 2026-07-13 (see issue #105 comment
for the crash log).

**Provenance:** Measured on Thor. The SM_110 trait in `lobes/machines/_traits.py`
defines `reranker enforce_eager=true` as a sm_110 characteristic. It is
inherited by the Thor `CardStrategy` (composes the trait).

### 4. `senses` multimodal gear is fully operational

The Spark profile's `senses` (the Gemma 4 12B multimodal gear) is served on
Thor at the same context (32K) and util (0.14) as on Spark. Live validation
(2026-07-13) confirmed the gear loads, answers requests, and compares vision
embeddings correctly.

### Note on future verification

- **Issue #106** tracks re-verifying the Triton-ATTN knobs on GB10 (Spark) to
  confirm they do not cause regressions there. They are expected to work (Spark
  can run FlashInfer fine, so Triton is not needed, but forcing it anyway
  should not break correctness).
- **Issue #109** tracks the FP8 KV cache issue on Thor (uncalibrated scale
  warnings). It remains open pending checkpoint re-calibration or a vLLM fix.
  The `auto` workaround is stable and correct but not ideal long-term.
- **Issue #107** tracks broader work to tuned-small-model defaults for every
  card, not just UNKNOWN. Spark and Thor are mature; this is future work.

## See also

- `lobes explain tuning` / `lobes explain profiles` — brief in-CLI reference
- `lobes explain machine` — machine profile flags and detection
- `docs/tuning-profiles.md` — workload (`purpose`) profiles and shahizat's
  benchmark baseline
- `docs/colleague-stack.md` — the fleet contract (roles, context, memory budget)
- `lobes/machines/` — CardStrategy modules (one per chip; the source of truth
  for detection signatures and machine knobs)
- `lobes/profiles/` — profile schema, loader, renderer, and built-in TOML
- `lobes/runtime/_detect.py` — the detection fact-gatherer
- Issue #105 — reranker `cudaErrorLaunchFailure` on Thor (fixed by
  enforce_eager + TRITON_ATTN)
- Issue #106 — GB10 re-verification of TRITON_ATTN knobs (pending)
- Issue #107 — broader tuned-small-model defaults for every card (future work)
- Issue #109 — uncalibrated FP8 KV cache scale warnings on Thor (workaround:
  `auto`)
