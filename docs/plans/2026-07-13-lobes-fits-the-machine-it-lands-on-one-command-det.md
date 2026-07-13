# Build Plan — lobes fits the machine it lands on: one command detects the card (GB10, Thor, Orin, Orin Nano Super) and serves a profile tuned for THAT box — not a lowest-common-denominator config that fits none

slug: `lobes-fits-the-machine-it-lands-on-one-command-det` · status: `exported` · from frame: `lobes-fits-the-machine-it-lands-on-one-command-det`

> lobes fits the machine it lands on: one command detects the card (GB10, Thor, Orin, Orin Nano Super) and serves a profile tuned for THAT box — not a lowest-common-denominator config that fits none

## Tasks

### t1 — Profile schema + built-in profiles (lobes/profiles/): a Profile declares, per role, whether it is feasible, which model serves it, and every machine knob (gpu util, max-model-len, quantization, kv-cache dtype, attention backend, enforce-eager, max-num-seqs). Ships spark.yaml (default, from today's template values) and thor.yaml (from the 4 edits validated this session). Loader resolves built-ins + operator-defined profiles and an explicit --profile override

- instruction: OWNS: lobes/profiles/ (new package: schema.py, loader, builtin/spark.yaml, builtin/thor.yaml) + tests/test_profiles.py. Touch NOTHING else. thor.yaml must encode the 4 divergences validated live this session: cortex kv_cache_dtype=auto; embedder attention_backend=TRITON_ATTN; reranker attention_backend=TRITON_ATTN + enforce_eager=true. spark.yaml must reproduce the CURRENT template values exactly (util 0.30/0.14/0.06/0.06, cortex 131072, senses 32768) so GB10 behaviour is unchanged.
- covers: c6, h11, c9, c10
- acceptance:
  - A Profile round-trips: load -> serialise -> identical; an unknown role or unknown knob is a load error, not a silent drop
  - thor.yaml encodes exactly the 4 validated Thor divergences (cortex kv_cache_dtype=auto; embedder+reranker attention_backend=TRITON_ATTN; reranker enforce_eager=true) and spark.yaml reproduces today's shipped template values byte-for-byte
  - An operator-defined profile in the deployment dir is discovered and can override a built-in of the same name; nothing in the loader mutates a profile at runtime

### t2 — Card detection (lobes/runtime/_detect.py): identify the host card by compute capability + device name + total memory, returning a named card id (gb10-spark, jetson-agx-thor, ...) or UNKNOWN. Must NOT read nvidia-smi memory fields — they report [N/A] on Thor's integrated GPU

- instruction: OWNS: lobes/runtime/_detect.py (new) + tests. Touch NOTHING else. Detect via device name + compute capability + total memory. HARD CONSTRAINT: nvidia-smi reports memory.used=[N/A] on Thor's integrated GPU — do not read nvidia-smi memory fields. /proc/device-tree/model yields 'NVIDIA Jetson AGX Thor Developer Kit' on this box; 'nvidia-smi --query-gpu=name,compute_cap' yields 'NVIDIA Thor'. No torch import (the CLI has no torch dep). UNKNOWN is a first-class result, never a silent Spark fallback.
- covers: c8, h1, h9, c4
- acceptance:
  - Detection distinguishes GB10 (sm_121) from Thor (sm_110) on real hardware, and returns UNKNOWN rather than guessing for a card it does not know
  - Detection never consults nvidia-smi memory.used/memory.total; a test asserts the code path does not depend on those fields (they are [N/A] on Thor)
  - Detection works without importing torch (the CLI has no torch dependency)

### t3 — Parameterise the compose templates so every machine-dependent knob flows from the profile via .env: per-gear kv-cache-dtype, attention-config, enforce-eager, gpu util, max-model-len, quantization, max-num-seqs, and the model per role. Includes deleting the dead MULTIMODAL_ATTENTION_BACKEND env (VLLM_ATTENTION_BACKEND no longer exists in the pinned nightly; --attention-config is the only honoured knob)

- instruction: OWNS: lobes/templates/ (fleet/docker-compose.yml, fleet/env.example, docker-compose.yml, env.example). Touch NO python. Ground truth for the Thor knobs is the working hand-edited /home/thor/.lobes/docker-compose.yml on this box — diff against it. Note --attention-config takes JSON and MUST be single-quoted in YAML ('--attention-config={"backend": "TRITON_ATTN"}') or compose fails to parse. Delete the dead MULTIMODAL_ATTENTION_BACKEND env: VLLM_ATTENTION_BACKEND no longer exists in the pinned nightly (0.23.1rc1.dev672).
- covers: c9, h2, c2, h7
- acceptance:
  - Every one of the 4 Thor divergences is expressible as a profile value with NO code fork per card — proven by rendering thor.yaml and diffing against the hand-edited ~/.lobes/docker-compose.yml that works today
  - Rendering spark.yaml produces the current shipped compose (no behaviour change for existing GB10 deployments)
  - The dead VLLM_ATTENTION_BACKEND/MULTIMODAL_ATTENTION_BACKEND env is gone; the attention backend is set via --attention-config only

### t4 — lobes init applies the resolved profile: detect card -> pick profile -> render .env + compose. Adds --profile to override, and REFUSES-OR-WARNS on an unrecognised card rather than silently applying the Spark profile

- instruction: OWNS: lobes/cli/_commands/init.py + the profile-resolution glue in lobes/cli/_runtime_ops.py. Coordinate with t6 if it also needs _runtime_ops. Keep the existing dry-run-by-default mutation-safety contract: init still requires --apply.
- depends on: t1, t2, t3
- covers: c1, h6, c5, c15, h12
- acceptance:
  - On a supported card, a bare 'lobes init' picks the right profile with no flags and needs zero hand-edits to the generated compose
  - 'lobes init --profile <name>' overrides detection, including forcing a profile onto a card it was not validated for (with a warning)
  - On an UNKNOWN card, init refuses (or warns loudly and requires an explicit --profile) — it never silently falls back to Spark

### t5 — lobes doctor / lobes status report the profile: which profile was chosen, which card was detected (name + compute capability + memory), and whether the profile is validated-on-this-card or forced

- instruction: OWNS: lobes/cli/_commands/doctor.py + status.py. Depends on t4's resolution glue — do not re-implement detection, call it.
- depends on: t4
- covers: c8, h9
- acceptance:
  - 'lobes doctor' names the detected card and the chosen profile, and flags a forced/unvalidated combination as a warning
  - The operator never needs to know the card's compute capability to get a correct boot — it is reported to them, not required from them

### t6 — Role feasibility is honoured end-to-end: a role the profile declares unavailable is omitted (or marked unserved) in 'lobes capabilities' and GET /capabilities, and the gateway 4xx's a request for it instead of silently routing it to a different model

- instruction: OWNS: lobes/gateway/ + the capabilities command. This extends the existing #92 invariant ('lobes never advertises a capability it cannot serve') to the HARDWARE dimension — reuse that machinery rather than inventing a parallel path.
- depends on: t1
- covers: c10, h3
- acceptance:
  - A profile with cortex unavailable produces capabilities that do NOT advertise cortex, and a POST with model=cortex returns 4xx rather than being served by another gear
  - No role is ever advertised that the card cannot serve (extends the #92 'never advertise what you cannot serve' invariant to the hardware dimension)

### t7 — Per-role CORRECTNESS probes (not just /health): cortex answers a known-answer question; embedder ranks a paraphrase above an unrelated string; reranker puts the relevant document first. A role that is healthy but semantically wrong must FAIL

- instruction: OWNS: lobes/assess.py (or a new lobes/probes.py) + lobes/cli/_commands/assess.py. Read-only verb; must not mutate the deployment. The three probes are known-answer generate, embed paraphrase-beats-unrelated, and rerank relevant-doc-ranks-first. On this Thor box today the rerank probe MUST FAIL and the embed probe MUST PASS — that is the acceptance test for the probes themselves.
- covers: c11, h4, h10
- acceptance:
  - The rerank ordering probe FAILS on this Thor box today (the relevant doc does not rank first) — the probe catches the real bug rather than being a test that only passes
  - The embed probe passes on Thor today (cos(paraphrase) > cos(unrelated)) and would have caught the FLASH_ATTN hang, which /health did not
  - Probes are wired into a read-only verb and never mutate the deployment

### t10 — Validate the Thor profile on the physical Thor: from an EMPTY deployment dir, 'lobes init && lobes serve' with zero hand-edits, then run every correctness probe and record the score. The reranker RUNS (serves requests without killing its engine) but is KNOWN-INCORRECT — its ordering probe is an expected failure tracked in #105/#106, which is where the fix lands. The profile neither hides that nor blocks on it

- instruction: Validation, run ON the Thor box. Move the current hand-edited ~/.lobes aside first (its 4 edits would mask a template regression), then init from empty. rerank STAYS SERVED: eager mode stops the engine-killing cudaErrorLaunchFailure, so it responds — it is simply not yet correct. Record the probe scores as evidence.
- depends on: t4, t6, t7
- covers: c7, h5, c18, h14, c3, h8
- acceptance:
  - Clean init+serve on Thor brings up 100% of the roles the thor profile claims — including reranker, which must SERVE (respond, no cudaErrorLaunchFailure) — with zero hand-edits to the generated compose
  - The rerank ordering probe is recorded as a KNOWN/EXPECTED failure referencing #105/#106: it neither passes silently nor blocks the profile work, and it flips to a hard failure once those issues close
  - The 1-of-4-on-first-boot baseline is reproduced with the stock Spark template on Thor, so the improvement is measured rather than asserted
  - Pointing the Spark profile at Thor still crashes (the 'fits none' cost is real), and the Thor profile is not applied to the GB10 where it would under-use it

### t11 — Document the profile surface: docs/machine-profiles.md (how detection resolves, how to write a profile, the knob reference), plus 'lobes explain profiles', and an honest support table — Spark (default, validated) + Thor (validated); Orin / Orin Nano Super named but UNVALIDATED. Thor's reranker is documented as served-but-known-incorrect (#105/#106), not quietly omitted

- instruction: OWNS: docs/machine-profiles.md (new), lobes/explain/, README/CLAUDE.md support table. Docs only, no source. Do not claim Orin support. Be explicit that Thor's rerank runs but is not yet correct.
- depends on: t10
- covers: c15, h12, c6, h11
- acceptance:
  - The support table states plainly which cards are validated and which are aspirational; no doc claims Orin support
  - Every knob in the reference traces to a human-validated observation on a real card (boot log, probe result, measured budget)
  - Thor's known-incorrect reranker is documented with a pointer to #105/#106 rather than being hidden

### t12 — Per-chip STRATEGY PATTERN (foundational — do this before the other tasks build on top of the current shape): one module per chip under lobes/machines/ (CardStrategy: its own detection signature + per-role knobs + provenance) plus a small shared registry. profiles.py / _detect.py / init.py stop carrying per-chip tables and derive from the registry instead. Nothing is deleted: MachineProfile, MACHINE_PROFILES, detect_machine() and their switch/benchmark callers keep working, rebuilt FROM the registry. Also re-cuts the false premise the spec originally carried (that lobes had no machine-profile axis at all — it does)

- instruction: OWNS: lobes/machines/ (new), and the registry-derivation edits to lobes/profiles.py + lobes/runtime/_detect.py. Foundational: land this BEFORE t1/t2/t4 build per-chip tables that would then have to be unpicked. Do NOT delete the legacy API (MachineProfile / MACHINE_PROFILES / detect_machine / resolve_serve_config) — derive it from the registry. stdlib only (dependencies = []); an explicit registry with imports in machines/__init__.py, no plugin/entry-point machinery.
- covers: c19, h15
- acceptance:
  - Adding a new chip = ONE new file + ONE registration line, proven by a test that registers a synthetic chip strategy and shows detection, profile resolution and knob rendering all pick it up with ZERO edits to profiles.py / _detect.py / init.py
  - Every pre-existing test passes UNMODIFIED — if an existing test's expectations must change, legacy behaviour was broken and the refactor has failed
  - The existing thor row (status='configured': flashinfer / 32768 / util 0.6 — an unvalidated guess that live Thor testing contradicts) is replaced with the measured values and marked load-tested; detect_machine()'s silent 'generic' fallback is preserved for its legacy callers but is NOT the source of truth for init

## Risks

- [follow_up] Reranker returns deterministically wrong orderings on Thor and MAY be wrong on the GB10 too (nothing checks ordering today). Tracked in lobes-cli#105 / #106. Until #106 answers, t8 cannot claim 4/4 on Thor — the thor profile must either fix rerank or declare it unavailable rather than advertise it broken. (task t8)
- [unknown_nonblocking] Jetson AGX Orin (sm_87, no NVFP4) and Orin Nano Super are UNVALIDATED — no physical board in hand. The schema must accommodate them (smaller cortex model, roles disabled) but no profile ships as supported until someone boots one.
- [unknown_nonblocking] Detection runs on the HOST (the lobes CLI), while the kernels that actually differ run INSIDE the container. If the two ever disagree about the card, the profile is resolved against the wrong truth. (task t2)
- [follow_up] DECISION (supersedes the framing of r1): on Thor the reranker stays SERVED and advertised — it runs, it is just not yet correct. Correctness is fixed in lobes-cli#105/#106, not by hiding the role. The correctness probe records it as a known failure until then. (task t10)
