# lobes fits the machine it lands on: one command detects the card (GB10, Thor, Orin, Orin Nano Super) and serves a profile tuned for THAT box — not a lowest-common-denominator config that fits none

> lobes fits the machine it lands on: one command detects the card (GB10, Thor, Orin, Orin Nano Super) and serves a profile tuned for THAT box — not a lowest-common-denominator config that fits none

## Audience

- Operators running lobes on a specific NVIDIA box — today DGX Spark GB10 and Jetson AGX Thor; next Jetson AGX Orin and Orin Nano Super — plus the lobes agent deployed on it.

## Before → After

- Before: lobes ALREADY has a machine-profile axis — lobes/profiles.py ships MachineProfile + MACHINE_PROFILES (spark/thor/blackwell/generic) and detect_machine(), wired to VLLM_MACHINE and used by switch/benchmark. But it is (a) ONE knob-set per machine, not per role; (b) missing the knobs that actually matter — kv-cache dtype, enforce-eager, model-per-role, role feasibility; (c) IGNORED by the fleet compose (the default path), which hardcodes DGX Spark GB10 values; (d) its thor row is an unvalidated GUESS (status 'configured': flashinfer / 32768 / util 0.6) that live testing on a physical Thor CONTRADICTS; and (e) detect_machine() silently falls back to 'generic' instead of admitting it does not know the card. On Thor (sm_110) the shipped fleet scored 1 of 4 roles correct on first boot; cortex needed --kv-cache-dtype=auto, embedder needed --attention-config TRITON_ATTN, reranker needed --enforce-eager and STILL returns wrong rankings.
- After: lobes detects the host card and applies a per-machine profile (memory budget, quantization, attention backend, KV dtype, context, eager-vs-cudagraph, which roles are even feasible), so a supported box boots correct-and-tuned on first try.

## Why it matters

- A single 'best for all machines' config fits none of them: it either wastes a big box's headroom or fails to boot on a small one.

## Requirements

- lobes detects the host card (compute capability + total memory + device name) and resolves a named profile; the operator can override the choice explicitly (--profile) and see what was chosen (lobes doctor / lobes status).
  - honesty: Detection is unambiguous on every target: GB10, Thor, AGX Orin and Orin Nano Super are each distinguishable from inside the container/host without guessing. (Note: nvidia-smi reports memory.used=[N/A] on Thor's integrated GPU, so detection cannot rely on nvidia-smi memory fields.)
- A profile owns the machine-dependent knobs, not just memory: GPU mem util per role, max-model-len, quantization, KV-cache dtype, attention backend, enforce-eager, and max-num-seqs.
  - honesty: The knob set is sufficient: every divergence found on Thor so far (kv-cache dtype, attention backend, enforce-eager) is expressible as a profile value — no code fork per card.
- A profile also declares which ROLES are feasible on that card and which model serves each. On Orin Nano Super the 27B NVFP4 cortex is not runnable at all, so the profile must select a smaller cortex model or declare cortex unavailable — capabilities must never advertise a role the box cannot serve.
  - honesty: A role can be declared unavailable end-to-end: lobes capabilities / GET /capabilities omit it (or mark it unserved), and the gateway 4xx's a request for it rather than silently routing to a different model.
- Every role ships a CORRECTNESS probe, not just a /health check: embed must rank a paraphrase above an unrelated string, rerank must put the relevant document first, cortex must answer a known-answer question. A role that is 'healthy' but semantically wrong must fail the probe.
  - honesty: The rerank probe FAILS today on this Thor box (relevant doc does not rank first) — i.e. the probe catches the bug we actually hit, rather than being a test that only passes.

## Honesty conditions

- One command does it: on a supported card, 'lobes init' picks the right profile with no flags, and 'lobes doctor' names which profile it picked and why (detected card + capability + memory).
- The 'fits none' cost is real and measurable: a single shared config either leaves a big box's headroom unused or fails to boot a small one — demonstrable by pointing the GB10 profile at Thor (it crashes) and a Thor-safe config at the GB10 (it under-uses it).
- The operator never has to know the card's compute capability to get a correct boot — the tool knows it; the operator only overrides when they WANT to.
- 'Correct on first try' is checked, not asserted: every role the profile claims passes its correctness probe on a clean boot, and a role the card cannot serve is absent from capabilities rather than broken-but-advertised.
- Every knob in a profile traces to a human-validated observation on a real card (a boot log, a probe result, a measured budget) — no value is there because a search loop landed on it, and lobes never mutates a profile at runtime.
- A profile can be proven on a real box: booting a supported profile from a clean ~/.lobes needs zero hand-edits to the generated compose file.
- lobes is honest about coverage: it ships Spark (default) + Thor as supported, names Orin/Orin Nano Super as unvalidated, and on an unrecognised card it refuses-or-warns rather than silently applying the Spark profile.
- The baseline is reproducible: a clean 'lobes init' + 'lobes serve' on Thor with the stock template reproduces 1/4-correct-on-first-boot, so 4/4-with-0-edits is a measured improvement, not a claim.
- The claim is checkable in the code as it stands: lobes/profiles.py really does define MachineProfile/MACHINE_PROFILES/detect_machine (grep it), the fleet compose really does hardcode the Spark values rather than read them, the thor row really is status='configured' with attention_backend='flashinfer', and detect_machine really does fall back to 'generic'. If any of those five is false, this before_state is wrong and the spec must be re-cut.
- Extensibility is PROVEN, not asserted: a test registers a synthetic new chip strategy and shows detection, profile resolution and knob rendering all pick it up with ZERO edits to profiles.py / _detect.py / init.py — and every pre-existing test still passes UNMODIFIED (if an existing test's expectations must change, legacy behaviour was broken and the refactor has failed).

## Success signals

- On a supported box, a clean 'lobes init && lobes serve' brings every role it claims up correct on the FIRST try — no hand-editing of docker-compose.yml. On an unsupported box lobes says so instead of silently mis-serving.
- Measurable: on each supported card, a clean 'lobes init && lobes serve' from an empty deployment dir brings up 100% of the roles the profile claims, each passing its correctness probe, with ZERO hand-edits to the generated compose. Measured baseline (Thor, this session, stock Spark-tuned template): 1 of 4 roles correct on first boot — senses came up clean; cortex crash-looped on an fp8-KV assert; embedder accepted requests and never answered; reranker killed its own engine with cudaErrorLaunchFailure. Reaching 3 of 4 took 4 hand-edits to docker-compose.yml. Target for the Thor profile: 4/4 with 0 edits (the 4th, rerank, is gated on #105/#106).

## Scope / boundaries

- Not a universal autotuner and not a benchmark-search: a profile is a hand-validated, checked-in config for a named card, not something lobes discovers at runtime by trying settings.
- Only cards we have ACTUALLY BOOTED ship as supported profiles: DGX Spark GB10 (the default) and Jetson AGX Thor (validated this session). Jetson AGX Orin and Orin Nano Super are named future targets — the design must accommodate them, but lobes does not promise them until someone boots one.

## Non-goals

- Not cross-vendor: no AMD/Intel/CPU-only targets. NVIDIA CUDA boxes only.

## Assumptions

- A profile is validated by RUNNING it on the physical card — a profile for a box nobody has booted is marked unvalidated, not shipped as supported.

## Decisions

- A machine profile pins the MODEL for each role, not just tuning knobs: the profile is the full machine contract (feasible roles + model per role + knobs), and the catalog is the menu it selects from. This is what lets Orin Nano Super downshift cortex to a small model instead of only disabling it.
- Profiles are default + overridable, and an operator may keep SEVERAL: lobes ships built-in profiles (Spark default, Thor), auto-detection picks one, --profile overrides it, and an operator can define their own profile rather than being limited to the shipped set.
- ARCHITECTURE: per-chip knowledge lives behind a STRATEGY PATTERN — one module per chip (lobes/machines/<chip>.py) owning its own detection signature, per-role knobs and provenance, plus a small shared registry. Adding a chip = one new file + one registration line; it must NOT mean editing shared tables in profiles.py / _detect.py / init.py, and a change for one chip must not be able to break another. Old code is NOT deleted: MachineProfile, MACHINE_PROFILES, detect_machine() and their switch/benchmark callers keep working, rebuilt FROM the registry rather than duplicated.

## Hard questions

- Does a profile pin exact MODELS per role (so Orin gets a smaller cortex), or only tuning knobs for the models the catalog already picks? The Orin Nano Super case forces model selection into the profile — which makes 'profile' and 'catalog' overlap. Which owns the model choice?

## Open / follow-up

- Is the reranker's wrong ordering Thor-specific, or a pre-existing lobes bug also live on the GB10? Nothing currently checks rerank ordering. TRACKED, not blocking this spec: agentculture/lobes-cli#105 (Thor: wrong rankings + cudaErrorLaunchFailure) and #106 (verify the same ordering probe on the GB10; the result decides whether the fix belongs in a Thor profile or in the shared rerank template).
