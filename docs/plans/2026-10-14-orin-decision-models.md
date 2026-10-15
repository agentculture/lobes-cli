# Build Plan — orin decision models

slug: `orin-decision-models` · status: `exported` · from frame: `orin-decision-models`

> The Jetson AGX Orin serves the mesh's decision models — Cloudflare Clef 27B (schema-scored decisions, Ampere-fit 4-bit) and, later, a fine-tuned Strands Decider 2B — through lobes, addressable from every mesh member

## Tasks

### t1 — Spike: load a 4-bit Clef + its joint schema head under transformers on the Orin, beside the live orin-embed shape

- instruction: Live-box task, not a coding task. Run in a scratch container on the Orin with a hard --memory cap and `HF_HUB_DISABLE_XET`=1 (zero-swap board; an uncapped download OOM-killed a lane before). Evict page cache before load (`posix_fadvise` DONTNEED). Try in order: JRamirez-UAB/clef-GPTQ-W4A16-g64-embed-int8 via transformers compressed-tensors, then a bitsandbytes NF4 build (Aikimi/clef-nf4 or jlancaster/clef-27b-nf4) — read each card first and verify the joint head files (`joint_head`.safetensors, `joint_schema_model.py`) load over the quantized backbone. Do NOT touch ~/.lobes or the live compose project. Never name non-fleet containers seen on any box in the transcript.
- covers: c4, h9
- acceptance:
  - docs/evidence/<date>-spike-clef-sidecar-orin.txt records: the exact checkpoint id + HF revision, its config.json `quant_method` (weight-only: W4A16/WNA16 or NF4 — never an FP4-activation path), the transformers/torch versions, resident memory, and min available host memory (>= 2 GiB) while orin-embed's three lanes keep answering
  - the transcript shows one /v1/systemone-shaped choice decision answered with per-option probabilities summing to 1 (±0.01) and its latency for a <= 4096-token state — or a NO-GO with the failing step and error text

### t2 — Add the decider role to lobes/roles.py: ROLES and every per-role table

- instruction: Touch only lobes/roles.py and its tests. Contract pinned for parallel tasks: role name 'decider', catalog `role_hint` 'decider', env prefix 'DECIDER', backend name 'decider', task family 'decide', path '/v1/systemone'. Mirror how 'associate' was added (git log -S associate -- lobes/roles.py).
- covers: c33, h15, c7, h10, c14
- acceptance:
  - lobes/roles.py ROLES contains 'decider'; `ROLE_RESPONSIBILITIES`\['decider'\] == (`typed_decision`, classification, `routing_choice`, triage, `guardrail_check`, `final_decision`) and `ROLE_FORBIDDEN`\['decider'\] == (`repo_action`, `security_decision`), pinned by a test in tests/`test_colleague_contract.py`
  - `ROLE_FORBIDDEN`\['hand'\] is unchanged (still forbids `final_decision`) and `ROLE_ROLE_HINT`\['decider'\] == 'decider'; the existing role tests pass

### t3 — Add catalog entries for the two decision checkpoints, with a new decide task and a sidecar engine value

- instruction: Touch only lobes/catalog.py and tests/`test_catalog`\*.py. Add a task value 'decide' and an engine constant for the decider sidecar (e.g. `ENGINE_DECIDER_SIDECAR` = 'decider-sidecar'). Pin `hf_revision` by resolving HEAD via the HF API on the day. Comments carry provenance (card URL) and say PUBLISHED-ELSEWHERE, not measured (#108). If t1 picked a different Clef build, use that one.
- covers: c4
- acceptance:
  - lobes/catalog.py has an entry for the chosen 4-bit Clef build (`role_hint`='decider', task='decide', quantization recorded, `hf_revision` pinned, status='configured') and one for StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 (`role_hint`='candidate' or 'decider', quantization='none', bf16, status='configured'); tests/`test_catalog.py`::`test_catalog_ids_are_unique` passes
  - neither entry is selectable by 'lobes switch' as a generate gear (`serves_with_vllm` is False for them) — a test asserts it

### t4 — Ship the decider sidecar: a container that serves one decision checkpoint on POST /v1/systemone plus GET /health

- instruction: Mirror lobes/`embed_sidecar`/ + Dockerfile.embed-st (stdlib/FastAPI server, uv pip install --system, never pip). The strands backend wraps the strands-decider package's model loader in-process rather than running its unauthenticated 'serve'. Bind 0.0.0.0 inside the container only; the compose file (t8a) decides exposure. Read the strands-decider server source to confirm one-checkpoint-per-process (c39/h17) and record the answer in the module docstring.
- covers: c3, c37
- acceptance:
  - lobes/`decider_sidecar`/ serves POST /v1/systemone forwarding the request body to the loaded model and returning its native response unchanged (Clef's and strands-decider's own shapes), and GET /health returning 200 only once the model is loaded — unit-tested with a fake model under tests/`test_decider_sidecar.py`
  - lobes/templates/fleet/Dockerfile.decider builds the sidecar for both backends selected by an env var (`DECIDER_BACKEND`=clef|strands); no EXPOSE-to-host or port publishing appears in it

### t5 — Gateway: route POST /v1/systemone by the X-Lobes-Model header to the decider domain, body byte-for-byte, never shed

- instruction: Files: lobes/gateway/server.py, `_routing.py`, `_pressure_policy.py`, `_config.py` (+ tests/`test_gateway_decider.py`). `_config.py` also gets decider's `OPT_IN_BACKENDS` / `FEASIBLE_ENV` (`DECIDER_FEASIBLE`) / `DECIDER_BASE_URL` entries — t8 must not touch `_config.py`. Domains are declared in `DECIDER_DOMAINS` (name=url, comma-separated, mirroring `HAND_LORA_MODULES` parsing in `_config.py`:626-670); 'general' is the default. Do NOT call `rewrite_model` on this path — the body never carries a model field.
- depends on: t2
- covers: c3, h8
- acceptance:
  - a gateway test sends POST /v1/systemone with X-Lobes-Model: decider:<domain> and asserts the backend receives the body byte-identical; a request with no header resolves to decider:general
  - an undeclared domain returns 404 `model_not_found`; no decider wired or reachable returns 404 `role_infeasible`; with swap > 75% and iowait > 50% every decider request is still forwarded (never 429) — tests in tests/`test_gateway_decider.py`
  - the route sits behind the existing bearer gate (`GATEWAY_API_KEY` covers /v1/\*)

### t6 — Readiness: probe decider lanes on their own health path

- instruction: File: lobes/gateway/`_readiness.py` (+ tests/`test_gateway_readiness.py`). Reuse `probe_backend_ready`'s `health_path` override (the innereye precedent, `_readiness.py`:153-158); decider lanes never probe /v1/models.
- depends on: t2
- covers: c36, h19
- acceptance:
  - a test wires a decider backend that 404s /v1/models but answers GET /health 200 and asserts it is reported ready:true; the same backend down reports ready:false

### t7 — Capabilities advert + mesh: announce decider domains as their own roster key, route by domain, refuse conflicts

- instruction: Files: lobes/gateway/`_mesh_wire.py`, `_mesh_roster.py`, `_mesh_routing.py` and the capabilities serializer in server.py (after t5 lands — same file). Advert goes through lobes.roles.`role_payload`, never a bare asdict. Forward only via the existing single-hop mesh forwarder; send the peer's served id, not a backend name (known 404 trap).
- depends on: t5
- covers: c34, h16, c40, h18
- acceptance:
  - GET /capabilities lists decider with a domains map (domain -> checkpoint, quant, ready) and the mesh announcement carries the same domains
  - a unit test announces decider:general from member A and decider:<x> from member B and asserts member C forwards each by domain (X-Lobes-Mesh-Member on the reply); two members announcing one domain with different fingerprints are refused, never silently picked

### t8 — Render decider lanes: profiles/shapes tables, compose template services (no host ports), up target

- instruction: Files: lobes/profiles/{shapes,schema,render,`shape_render`}.py, lobes/cli/`_commands`/up.py, lobes/templates/fleet/docker-compose\*.yml (or a generated override for per-domain services), env.example. Service list is data (`DECIDER_DOMAINS`), so per-domain services belong in the generated docker-compose.shape.yml, not a static template. Every up carries --no-deps; give no service a `depends_on` on decider.
- depends on: t2, t3
- covers: c14, c37, h20
- acceptance:
  - 'decider' is in `OPT_IN_CORE_ROLES` (lobes/profiles/shapes.py) and the schema/render/`shape_render`/up `ROLE_SERVICE` tables; machine-as-brain renders nothing for it and an unwired decider renders `DECIDER_FEASIBLE`=false
  - each declared domain renders a decider-<domain> service from lobes/templates/fleet/Dockerfile.decider with no 'ports:' mapping — asserted by a golden/shape test

### t17 — Interim (d4): serve Cloudflare's original bf16 Clef as decider:general on the Spark through the decider sidecar

- instruction: Spark compose is hand-kept: add the decider-general service and `DECIDER_DOMAINS` via docker-compose.override.yml, never re-scaffold (memory: spark-compose-lacks-fingerprint-passthroughs). Use the lobes-deploy skill. Stopping the experiment container needs the user's go at that moment.
- acceptance:
  - a decider-general sidecar (`DECIDER_BACKEND`=clef, `DECIDER_MODEL`=Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c, Cloudflare's own `joint_schema_model.py`) answers POST /v1/systemone through the Spark gateway and, via the mesh, from the Orin and Thor gateways; per-question probabilities sum to 1 (±0.01)
  - docs/evidence/<date>-accept-clef-bf16-interim-spark.txt records the memory freed beforehand (only numbers, never names of non-fleet containers), resident memory, latency at <=4096 tokens, and the ~/.lobes backup taken before the change

### t18 — Build a calibration set and a disjoint labelled probe set of Clef-format decision records; publish privately to culture-dev

- instruction: Follow the hf-culture-dev-org convention (private first, pin upstream by revision, cite commit sha). The labelled probe set is what t11/t21 measure top-1 accuracy on; the calibration set must never be used for evaluation.
- acceptance:
  - two private culture-dev datasets (pinned by revision): a calibration set (>=256 Clef-format records: state + typed questions, mixed choice/noul/score, English) and a labelled probe set (>=100 records with ground-truth answers) with zero overlapping records — a test or script asserts disjointness
  - every record's source is cited (public dataset id + license, or hand-written by the mesh); no private mesh data and no names of non-fleet models

### t19 — Quantize the Orin build: W4A16 INT4 GPTQ from Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c with llm-compressor; upload privately to culture-dev

- instruction: Run the quantization on spark2 or on the Spark after the interim is up (needs ~55 GB for the bf16 source). Calibrate with t18's calibration set only.
- depends on: t18
- acceptance:
  - a private culture-dev model repo holds a compressed-tensors W4A16 (int4, symmetric, group 64) build of Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c with the joint head, vision tower, `lm_head`, GatedDeltaNet `in_proj_a`/b, norms and `embed_tokens` kept bf16 (no vLLM patch needed), plus Cloudflare's LICENSE/NOTICE, the exact llm-compressor recipe and versions, and the source revision in its card
  - the build loads in vLLM >= 0.31 on the Orin (`sm_87`) through a Marlin W4A16 kernel — the log line is quoted in the evidence

### t20 — Quantize the Blackwell build: NVFP4 (W4A4) from Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c with llm-compressor; upload privately to culture-dev

- instruction: Calibrate with t18's calibration set only. It is not expected to run on the Orin — do not claim it does.
- depends on: t18
- acceptance:
  - a private culture-dev model repo holds a compressed-tensors NVFP4 (W4A4) build of Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c with the same bf16 keep-list as the Orin build, LICENSE/NOTICE, recipe, versions and source revision in its card
  - the build loads in vLLM on a Blackwell box (Spark GB10 or Thor) with an FP4 (non-Marlin) kernel — the log line is quoted in the evidence

### t21 — Decider sidecar: a clef-vllm backend (vLLM 0.31 pooling runner in-process + Cloudflare's joint head) and its image variant

- instruction: Third-party wrappers (e.g. `clef_vllm.py` from other repos) may be read for the approach but not copied or executed; write our own against Cloudflare's code.
- acceptance:
  - lobes/`decider_sidecar` gains `DECIDER_BACKEND`=clef-vllm: the backbone runs in vLLM's pooling runner, Cloudflare's `joint_schema_model.py` head scores options in bf16, and POST /v1/systemone returns the same native response shape as the clef backend — unit-tested with fakes (no vllm import in tests)
  - lobes/templates/fleet/Dockerfile.decider-vllm builds it on a pinned vllm/vllm-openai:v0.31.0 digest (arm64 present), uv pip only, no host ports

### t22 — Catalog: entries for Cloudflare/clef bf16 (interim) and our two culture-dev builds; retire the third-party Clef quant entries to candidate status

- instruction: Same conventions as t3 (decide task, decider-sidecar engine, PUBLISHED-ELSEWHERE vs measured wording).
- depends on: t19, t20
- acceptance:
  - lobes/catalog.py lists Cloudflare/clef@2f3de3dd85f379784083b0814d997ab627200f0c (bf16, decider), the culture-dev W4A16 build (Orin) and NVFP4 build (Blackwell) with pinned revisions; the JRamirez entry is demoted with a comment citing d4; tests pass

### t23 — Verify both builds against the bf16 interim on the labelled probe set

- instruction: The bf16 reference is the t17 interim on the Spark.
- depends on: t17, t19, t20, t21
- acceptance:
  - docs/evidence/<date>-verify-clef-quants.txt reports, on t18's labelled probe set: top-1 accuracy for bf16, W4A16 (Orin) and NVFP4 (Blackwell); top-1 agreement and mean/max KL of each quant vs bf16; latency and resident memory per box
  - a build is promoted only if its top-1 agreement with bf16 on the labelled probe set is >= 97% (user-set threshold, 2026-10-14); a build below it is reported as such and not promoted; making either culture-dev repo public is a separate user decision recorded in the transcript

### t9 — Orin shape orin-decider (orin-embed + decider:general) and the exclusive heavy-lobe group; regenerate goldens

- instruction: Copy orin-embed.toml and add decider; budgets stay DECLARED until t11 measures them (#108 wording). Run goldens regen last, after every render change has landed.
- depends on: t8, t22
- covers: c9, h11, h12
- acceptance:
  - lobes/profiles/`builtin_shapes`/orin-decider.toml hosts embedder, reranker, the gemma2-embed lane and decider; orin.toml \[\[`exclusive_roles`\]\] lists it, and tests/`test_init_coresidency.py` refuses it alongside orin-cortex/orin-lobe/orin-associate
  - uv run python tests/goldens/regen.py regenerates every golden with the decider line and 'uv run pytest -n auto' passes

### t10 — Docs: decider role page, colleague-stack justification, rollback procedure, a lobes route worked example

- instruction: Every claim not yet measured is marked DECLARED/UNVALIDATED (#108). Do not name experimental/side models seen on any box.
- depends on: t5, t6, t7, t9
- covers: c21, h2, c14
- acceptance:
  - docs/decider.md covers both checkpoints, the quant answer for each (Clef: weight-only 4-bit on Ampere; Strands 2B: bf16, no quant), the X-Lobes-Model header, the pass-through response formats, the no-shed rule, and the Orin backup-restore rollback (never re-render alone)
  - docs/colleague-stack.md records why decider earns an irreversible role, and docs/decider.md shows lobes route's gear choice as a concrete /v1/systemone choice question (the named consumer for h2)
  - CLAUDE.md's role list, docs/openai-api.md and 'lobes explain' mention decider; markdownlint-cli2 passes on every touched doc

### t11 — Live Orin acceptance: back up ~/.lobes, render orin-decider, serve Clef as decider:general beside orin-embed, measure on a labeled probe set

- instruction: Use the lobes-deploy skill (scripts/lobes-compose.sh); never run compose in lobes/templates/. Boot order: decider first, then the gears, if the page-cache/util refusal appears. Gateway built from a TestPyPI dev wheel of this branch. Never name non-fleet containers in the transcript.
- depends on: t1, t10, t4, t21, t23
- covers: c1, h1, c25, h23, c23, h4, c38, h21, c24, h5, h9, h20
- acceptance:
  - docs/evidence/<date>-accept-orin-decider.txt records the ~/.lobes backup path taken BEFORE 'lobes init --shape orin-decider --apply', and the four lanes (embedder, reranker, gemma2-embed, decider) all answering through the gateway afterwards
  - on a fixed labeled probe set (committed, with ground-truth answers): per-question probabilities sum to 1 (±0.01), top-1 accuracy is recorded beside the quant card's 65/68 bf16 agreement, a confidence-vs-correctness table shows whether thresholds are usable, median latency for <= 4096-token states is recorded, and min available host memory stays >= 2 GiB; no bf16 Clef is booted
  - the decider port is refused from another host while the gateway route answers; GET /capabilities on another mesh member shows decider:general hosted by the Orin

### t12 — Live second-box acceptance: serve the stock Strands Decider 2B bf16 on spark2 as a decider domain, reached cross-box from every member

- instruction: spark2's login user is spark2, sudo needs a password, no Tailscale SSH. Pick a neutral domain name for the stock checkpoint (e.g. 'strands-general'); record the choice.
- depends on: t11
- covers: c26, h7, h1
- acceptance:
  - docs/evidence/<date>-accept-decider-2b-spark2.txt shows StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 served bf16 on spark2 as decider:<domain> and answered via X-Lobes-Model from the Orin, the Spark and the Thor gateways, each reply carrying X-Lobes-Mesh-Member: spark2
  - the stock checkpoint is used — no retrained adapter is required for this task (retraining is the jev-factory session's output, c18)

### t13 — Orin main path: serve our culture-dev W4A16 Clef through the clef-vllm sidecar backend on vLLM 0.31; A/B against the t1 transformers NF4 baseline

- instruction: Serve our t19 culture-dev W4A16 build through the t21 clef-vllm sidecar backend (Dockerfile.decider-vllm on pinned vllm/vllm-openai:v0.31.0). No third-party patch or wrapper is applied or executed (d4); the bf16 `embed_tokens` keep-list is what removes the need for an embed-quant patch. A/B against the t1 NF4 transformers baseline numbers and the t23 probe-set accuracy.
- depends on: t11, t23, t21
- covers: c4
- acceptance:
  - docs/evidence/<date>-ab-clef-vllm-vs-sidecar-orin.txt reports, on t18's labelled probe set, median latency, top-1 accuracy and resident memory for both paths; the vLLM path is adopted only if its top-1 agreement with bf16 is >= 97% (the t23 threshold)
  - the vLLM image used is recorded by digest and its vLLM version (>= 0.31.0) and an aarch64 + `sm_87` build is confirmed — or the task records a NO-GO with the failing step

### t15 — /validate-delivery: run every confirmed obligation's behavioral test agent-side and file evidence + behavioral deltas

- instruction: Run the /validate-delivery skill after waves 0-5 have merged into the branch and before /summarize-delivery. Tests run agent-side (uv run pytest for o1-o11, the t11/t12 transcripts for o12-o14); the devague CLI only records results.
- depends on: t12, t13
- covers: h5
- acceptance:
  - every confirmed obligation (o1-o14) has a filed 'devague evidence' record citing the test or live transcript that exercised it, with its pass/fail/partial outcome — a failing or partial outcome is filed as such, never suppressed
  - every behavior the run added, amended or removed beyond the obligations (e.g. a t13 NO-GO leaving the sidecar as the served path) is filed as a 'devague delta'

### t16 — /summarize-delivery: write the planned-vs-actual closure artifact

- instruction: Run the /summarize-delivery skill on complete, partial or failed outcomes alike — report a failure faithfully.
- depends on: t15
- acceptance:
  - docs/deliveries/<date>-orin-decision-models.md exists, written by the /summarize-delivery skill: planned vs actual per task, every approved deviation, evidence-backed delivery claims only (no claim without a filed evidence record), and the remaining work (e.g. retrained 2B domains from the jev-factory session, the t13 outcome)
  - the artifact names no experimental/side model seen on any box; markdownlint-cli2 passes on it

### t14 — PR: version bump, CHANGELOG, scope checks, evidence + delivery summary linked, review

- instruction: Use the cicd skill to open and shepherd the PR; reply to and resolve every review thread before merge.
- depends on: t16
- covers: c18, h13, c22, h3
- acceptance:
  - the PR diff contains no training/jev-factory code (grep for recipe.sh, training/, unsloth) and the version is bumped with a CHANGELOG entry (python3 .claude/skills/version-bump/scripts/bump.py minor)
  - the PR description cites the before-state check (no systemone/decider surface on main as of 2026-10-14) and links every evidence transcript; uv run afi cli doctor . --strict passes
  - the PR description links the delivery summary from t16 and the evidence records from t15

## Risks

- [unknown_nonblocking] An aarch64 + `sm_87` vLLM >= 0.31 image may not exist; the faster path (t13) can end as a NO-GO, leaving the transformers sidecar as the served path (task t13)
- [unknown_nonblocking] If t1 is a NO-GO for every 4-bit build, Clef cannot be served on the Orin at all; the plan stops at t1 and goes back to the user (the 2B path, t12, does not depend on Clef's quant) (task t1)
- [unknown_nonblocking] Clef + orin-embed may not fit together under the 2 GiB floor; c16 makes co-residency conditional on the measurement — a misfit reopens the shape decision with the user, not a silent gear drop (task t11)
