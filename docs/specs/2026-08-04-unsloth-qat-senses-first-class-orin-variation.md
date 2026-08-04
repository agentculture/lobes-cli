# unsloth QAT senses + first-class orin variation

> senses serves unsloth/gemma-4-12B-it-qat-w4a16, live-validated on this Jetson AGX Orin as a first-class orin variation, while the thor/spark machine profiles stay intact so moving the setup to another architecture stays a profile pick, not a rework
> instruction: verify end-to-end: lobes status shows senses loaded with the unsloth model id; curl the gateway model=senses known-answer + vision probes; confirm docs/evidence/ transcript committed

## Audience

- the operator of this Orin box and every Colleague-facing caller that addresses model=senses/multimodal through a lobes gateway; downstream bystanders: the thor/spark deployments, which must stay byte-identical

## Before → After

- Before: today this box serves senses = coolthor via a hand-maintained OPERATOR profile: detection resolves UNKNOWN -> base (which vetoes senses), thor-lobe's Thor-measured overrides clobber the Orin-measured util so .env is hand-edited (0.45), the csv-mode runtime:nvidia edits revert on every re-init, and the checkpoint's video/audio/reasoning story has never been probed here
- After: this box detects as orin, renders the orin-lobe shape over the builtin orin profile, and serves senses = unsloth/gemma-4-12B-it-qat-w4a16 at a live-measured budget (context per the model card, re-measured); image/video/audio/reasoning are each probed and honestly reported; thor/spark render byte-identical to before

## Why it matters

- portability is the point: moving this setup to another architecture becomes a profile pick instead of a rework of hand edits, and the fleet's first Ampere box gets reproducible, evidence-backed senses capabilities instead of a deployment only this box's .env remembers

## Requirements

- lobes/catalog.py gains a SupportedModel entry for unsloth/gemma-4-12B-it-qat-w4a16 modeled on the coolthor/gemma-4-12B-it-NVFP4A16 gear entry: `role_hint`=multimodal, `tool_parser`=gemma4 (runtime.`_parser` returns gemma4 for gemma-4\* ids), quantization=compressed-tensors (the HF card declares compressed-tensors QAT W4A16, explicitly vLLM-targeted), plus a per-model doc page per the doc= convention
  - instruction: add SupportedModel entry mirroring the coolthor gear: `role_hint`=multimodal, `tool_parser`=gemma4, quantization=compressed-tensors, doc page under docs/; read config.json first for context/modality literals
  - honesty: the catalog entry states only what the checkpoint's own config.json declares (`quant_method`, modalities, context) plus what live probes proved — no capability copied from marketing text
- W4A16 weight-only quantization is what makes this checkpoint plausible on Ampere `sm_87` (activations stay 16-bit, no Blackwell FP4 tensor cores needed) — but the unsloth export is compressed-tensors INT4 pack-quantized, a DIFFERENT kernel path than the coolthor NVFP4A16 (FP4) precedent this box actually booted; the int4 path is unproven on this box and the boot is the test
  - instruction: boot is the test; if vLLM refuses the quant path on `sm_87`, that refusal is recorded as the finding
  - honesty: the W4A16-fits-`sm_87` claim is proven only by the boot itself — the coolthor precedent makes it plausible, the unsloth checkpoint's own weights loading and serving on this box makes it true
- the orin variation becomes first-class: a builtin lobes/profiles/builtin/orin.toml plus a lobes/machines/orin.py CardStrategy (per `_registry.py`: adding a chip is one new module + one register line; detection reads nvidia-smi name/`compute_cap`, Jetson device-tree model string, hostname — today Orin resolves UNKNOWN -> base.toml, which vetoes senses). The #108 earn-by-booting bar is met by the 2026-07-17 live validation in docs/orin-profiles.md
  - instruction: add lobes/machines/orin.py CardStrategy + register line (signature: nvidia-smi name/`compute_cap` 8.7 + device-tree Jetson AGX Orin) + lobes/profiles/builtin/orin.toml with the live-measured senses knobs
  - honesty: detection resolves THIS box to orin (verified live via lobes doctor/whoami on this hardware) and an unrecognised card still falls to base — no other card's resolution changes
- goldens expand automatically: tests/`test_profile_goldens.py` and tests/`test_shape_goldens.py` parametrize over `builtin_names`(), so a builtin orin profile requires orin.env plus one golden per shape x orin pairing, regenerated via tests/goldens/regen.py
  - instruction: run regen.py, then uv run pytest -n auto; inspect git diff --stat on tests/goldens/
  - honesty: goldens are REGENERATED via tests/goldens/regen.py, never hand-typed, and the diff shows only orin-related additions — zero byte changes to existing base/spark/thor goldens
- the senses lane's --speculative-config (google/gemma-4-12B-it-assistant MTP draft) is a hardcoded literal in lobes/templates/fleet/docker-compose.yml, byte-guarded against the coolthor catalog entry by tests/`test_catalog.py`, and RoleProfile (lobes/profiles/schema.py) has no speculative knob — swapping the senses checkpoint requires parameterizing it (env var with default) or updating it in lockstep
  - instruction: parameterize the senses --speculative-config as `MULTIMODAL_SPECULATIVE_CONFIG` with the current literal as default + an off sentinel; update the `test_catalog.py` byte-guard accordingly
  - honesty: with no env override the rendered senses lane is byte-identical to today's (the coolthor speculative-config literal remains the default) and an explicit off value omits the flag entirely — proven by the template-defaults golden and a new test
- thor-lobe's \[overrides.senses\] (`gpu_mem_util`=0.30 / `max_model_len`=131072, Thor-measured) overlay the card profile at render time and clobbered Orin's measured 0.45 — today patched by hand-editing `MULTIMODAL_GPU_MEM_UTIL` in .env (docs/orin-profiles.md). The first-class orin variation needs a structural answer: an orin-fit shape, or a precedence/override rule, not a hand edit
  - instruction: add lobes/profiles/`builtin_shapes`/orin-lobe.toml (hosts senses+embedder+reranker; overrides.senses from the live boot measurements); regen goldens
  - honesty: thor-lobe stays byte-identical for thor (its goldens unchanged); orin-lobe's senses overrides carry ONLY live-measured Orin values, and the shape hosts no stt/tts
- live-test means: boot the new checkpoint on this box, measure the accepted util/KV pool, then pass the senses probes lobes assess encodes (known-answer via the gateway alias, vision red-image intake) — end state is this box's deployment serving `MULTIMODAL_MODEL`=unsloth/gemma-4-12B-it-qat-w4a16 through the orin variation, matching the probes docs/orin-profiles.md records for the incumbent
  - instruction: boot senses, record accepted util/`max_model_len`/KV pool from vLLM logs, run the assess known-answer + vision probes, then the video/audio/reasoning probes; write docs/evidence/2026-08-04-accept-senses-unsloth-orin.txt
  - honesty: probes run on THIS box against the NEW checkpoint through the gateway alias — reusing the 2026-07-17 coolthor transcript as evidence for the unsloth checkpoint would be fabrication
- activation must survive re-render: this box's NVIDIA toolkit runs csv mode, so the deployed compose files carry hand-edited runtime:nvidia stanzas that a re-init REVERTS (docs/orin-profiles.md divergence 1, flagged there as needing a template knob or machine-strategy overlay) — the orin variation either lands that knob in-repo or documents re-applying the edit
  - instruction: add a template knob or machine-strategy overlay emitting runtime:nvidia for csv-mode boards (detected or profile-declared); verify by re-rendering on this box and booting
  - honesty: after the change, a fresh render on this csv-mode box boots WITHOUT hand-editing GPU stanzas, or the manual step is documented as a numbered known divergence — silence is the failure mode
- audio intake is a user-stated goal, but it is gated on vLLM, not the checkpoint: issue #101 found `gemma4_unified` silently drops `input_audio` (200 OK, fluent answer that ignored the audio) even though the config declares audio support — the live-test must re-probe audio on the pinned nightly, and if still dropped, audio stays honestly unserved (stt remains the speech path) until vLLM fixes it
  - instruction: re-run the #101 audio probe (prompt-token-count delta + content assertion on a known utterance) against the new checkpoint on the pinned nightly
  - honesty: the audio probe must be able to DETECT the silent drop (#101's signature: 200 OK with the audio ignored) — a fluent reply alone proves nothing; outcome reported either way, doc updated to match
- the swap's capability goal (user-stated 2026-08-04): the unsloth QAT checkpoint should give senses image, video, audio, and reasoning — each capability claim is honest only if probed live on this box, not inferred from the HF card
  - instruction: build the four-probe acceptance matrix into the live-test; record per-capability verdicts in the evidence transcript and mirror them into the catalog entry + doc page
  - honesty: each capability is claimed only from its own live probe with a negative control where one exists (vision: wrong-colour; video: reversed-motion; audio: silent-drop detection; reasoning: thought-trace present in `reasoning_content`) — the model card is the guide for what to ATTEMPT, never the evidence
- config.json VERIFIED (fetched unauthenticated 2026-08-04, not a gated repo): `text_config`.`max_position_embeddings`=262144 — the 256K claim is the checkpoint's own declaration, double the incumbent's 131072; `video_token_id`=258884 and `audio_config` both present (video is DECLARED natively — the repo's text+image+audio Gemma capability line predates this export); architecture Gemma4UnifiedForConditionalGeneration (the existing Dockerfile.vllm-gemma4 image serves this class); quantization = compressed-tensors INT4 pack-quantized symmetric, NOT FP4
  - honesty: wired knobs are read from the DOWNLOADED checkpoint's config.json at implementation time, not from this network fetch or the card prose
- playbook ordering (docs/model-switch-playbook.md par.1): benchmark the INCUMBENT coolthor on THIS box on the current engine BEFORE the swap — TTFT, decode tok/s from usage.`completion_tokens` (NEVER SSE chunk counts: the chunk-count trap under-reported 2x on 2026-07-31), MTP acceptance from the SpecDecoding engine logs, KV pool from the boot log — across short/medium/long request shapes; that baseline is unrecoverable after the swap
  - honesty: the pre-swap transcript records the incumbent's metric table (TTFT / `completion_tokens` decode / MTP acceptance / KV pool) on the current engine — any post-swap comparison cites it, and a comparison against numbers from a different engine build is called out as uncontrolled
- activation must PERSIST the Tegra iowait fix: the deployed .env still reads `LOBES_IOWAIT_DEGRADED_THRESHOLD`=50 (verified live 2026-08-04) while the running gateway survives only on an ephemeral shell-env override of 100 — this box's sugov kthreads inflate iowait to ~59% with zero disk I/O, so ANY compose recreate at threshold 50 resurrects indefinite 429-shedding of senses and would falsely fail the live-test; the orin variation must render or document the raised threshold (verify the exact env override name against lobes/gateway/`_pressure_policy.py` at implementation)
  - honesty: proven by a compose recreate at the persisted threshold: senses answers through the gateway (no 429 shed) while /proc/stat still reports the inflated Tegra iowait
- rollback is a snapshot, and it must exist BEFORE the swap: back up ~/.lobes (.env + both compose files) prior to re-render; restore = copy back + compose up (the coolthor weights stay in the HF cache — 7.7 GiB, disk has 1.6T free, verified). Re-init preserves .env per-key (`_apply_profile_env` merges into the existing file — operator lines like `PRIMARY_PEER_API_KEY`/`AUDIO_URL` survive) but pre-existing compose files need --force, which reverts hand edits
  - honesty: the snapshot's existence is verified BEFORE the swap step runs; restore is byte-for-byte (diff the restored files against the snapshot)
- the activated orin deployment also proxies the two heavy roles it does not host (user statement 2026-08-04): worker -> the Jetson AGX Thor (NEW wiring on this box: `WORKER_PEER_ORIGIN` pointing at thor's gateway :8000 + `WORKER_PEER_PROXY`=true + `WORKER_PEER_API_KEY` = thor's inbound `GATEWAY_API_KEY`, operator-provided per the O(machines) key model; functional since the 0.54.8 server.py fix) and cortex -> the DGX Spark (the existing `PRIMARY_PEER_`\* wiring, preserved) — so a caller on this box's gateway reaches senses locally, cortex from the Spark, and worker from the Thor
  - honesty: post-activation, model=worker and model=cortex requests through this box's gateway both answer 200 with X-Lobes-Proxied-By naming the thor/spark origin respectively — a 404 `role_infeasible` on worker means the wiring is absent, not proxied

## Honesty conditions

- the announcement is claimable only when this box actually serves the unsloth checkpoint via the orin variation AND the acceptance evidence exists under docs/evidence/ — a rendered .env alone is not shipped
- proven by diff: zero byte changes to spark.toml, thor.toml, their goldens, and the `test_profile_schema` pins — reviewable in the PR
- after re-render, .env still carries the `PRIMARY_PEER_`\* trio and a model=cortex request through this box's gateway still answers from the Spark with X-Lobes-Proxied-By
- reasoning is validated on THIS 12B checkpoint only when a live thinking request returns the trace in whichever field this vLLM build uses — playbook par.4: this build returns 'reasoning', and a probe reading only '`reasoning_content`' produces a FALSE stopped-thinking verdict — so the probe dumps sorted(message.keys()) and reconciles usage.`completion_tokens` against the visible field lengths before any verdict
- no caller-visible contract changes: model=senses/multimodal aliases, capabilities fields, and endpoint paths all answer exactly as before the swap
- the before-state is cited from docs/orin-profiles.md and the live .env read on 2026-08-04 — if implementation finds reality diverged, the frame is amended, not steamrolled
- every element is command-checkable: detection via lobes doctor/whoami on this box, the rendered .env against the orin-lobe golden, probes via assess, thor/spark via unchanged goldens
- the portability claim is demonstrated, not asserted: the zero-hand-edit reproduce command in the success signal is its test
- every listed signal is command-checkable and recorded in the evidence transcript; a failed signal is reported as failed, never silently dropped from the list
- reviewable in the PR: the committed transcript contains placeholders, never a literal key — same convention as the 2026-07-16 evidence file

## Success signals

- lobes init --profile orin --shape orin-lobe --apply reproduces the deployment on this box with ZERO hand edits; senses answers the assess probes (known-answer + vision) on the new checkpoint through the gateway alias; an acceptance transcript lands under docs/evidence/; the full pytest suite including regenerated goldens is green; thor/spark goldens unchanged

## Scope / boundaries

- thor.toml and spark.toml stay intact as the per-card portability mechanism (the user's explicit constraint): tests/`test_profile_schema.py` pins spark senses model == coolthor/gemma-4-12B-it-NVFP4A16 and thor senses identical to spark (exactly-four-divergences contract) — those pins change only if the user decides the fleet-wide senses checkpoint moves too
- cortex stays infeasible on Orin (the 27B primary quantizes activations to FP4 — modelopt/compressed-tensors W4A4 needs Blackwell tensor cores; a hard architecture line per docs/orin-profiles.md) and this box's cortex proxy wiring to the Spark (`PRIMARY_PEER_ORIGIN`/PROXY/`API_KEY` in ~/.lobes/.env) is untouched by the senses swap
- the acceptance evidence transcript redacts key material — `GATEWAY_API_KEY`/peer-key values appear as <...key> placeholders, following the recorded precedent in docs/evidence/2026-07-16-proxy-lobes-live-spark-thor.txt

## Non-goals

- the builtin orin-small shape stays DECLARED/UNVALIDATED and is not this deployment's shape — the assignment on this box is Orin hosts senses (user decision on #127, recorded in docs/orin-profiles.md Shape choice); nothing here validates or activates orin-small

## Assumptions

- budget hypothesis for unsloth-QAT-on-Orin: util starts from the coolthor-measured 0.45 (same 12B W4A16 size class) and `max_model_len` starts from the CARD-declared 256K per c18 — both re-MEASURED on the live boot, measured-not-computed (Orin's own 0.30 was refused live; thor-muse's 0.40 likewise); coolthor's 802,644-token KV pool at 0.45 suggests 256K/request leaves ~3x concurrency if the window holds
- reasoning rides machinery already wired: the senses lane hardcodes the gemma4 parser PAIR (tool parser + reasoning parser that consumes Gemma's <|channel>thought markers) — validated live on the 31B muse lane 2026-07-17, inherited by 12B lanes as a family rule but UNVALIDATED there (#108); the Orin live-test can be the validation
  - instruction: one thinking-mode request through the gateway; assert `reasoning_content` non-empty and content free of channel markers

## Scope exploration

- `s1` — `lobes/catalog.py (coolthor gemma-4-12B gear entry + #101 audio note)`: the coolthor entry is the template for the new gear: `role_hint`=multimodal, shape=unified text+image+audio, `tool_parser`=gemma4 (matches runtime.`_parser`'s gemma-4\* rule), quantization=compressed-tensors, `speculative_config` pinned to the google assistant MTP draft; its comments also record the #101 audio-dropped gap and the 2026-07-17 pythonic->gemma4 parser correction
  - seeds: `c2`, `c12`
- `s2` — `HF model card: unsloth/gemma-4-12B-it-qat-w4a16`: checkpoint exists: QAT (quantization-aware trained) W4A16, compressed-tensors serialization explicitly targeting vLLM, finetuned from google/gemma-4-12B-it, multimodal with vision encoder + declared audio; no MTP/draft head mentioned; card summary claims 256K context, which CONFLICTS with the repo-measured 131072 native for the 12B IT line — config.json must arbitrate
  - seeds: `c2`, `c3`
- `s3` — `lobes/profiles/builtin/spark.toml + thor.toml + base.toml`: both validated profiles pin senses = coolthor/gemma-4-12B-it-NVFP4A16 with util 0.14 / 32K (duo trim); thor is deliberately byte-identical to spark except four validated `sm_110` divergences overlaid from the machines registry; base.toml (what an undetected Orin resolves to today) vetoes senses outright
  - seeds: `c11`, `c4`
- `s4` — `tests/test_profile_schema.py`: pins spark senses model/knobs literally and enforces thor-encodes-exactly-the-four-validated-divergences — a fleet-wide senses checkpoint swap must edit these asserts; an orin-only pin leaves them untouched
  - seeds: `c11`
- `s5` — `lobes/machines/_registry.py + lobes/runtime/_detect.py`: adding a card = one new module + one register() line, registration order = detection precedence; detection probes nvidia-smi (name+`compute_cap` only, never memory), /proc/meminfo, Jetson /proc/device-tree/model, hostname — no orin strategy exists, so this box resolves UNKNOWN -> base profile
  - seeds: `c4`
- `s6` — `tests/test_profile_goldens.py + tests/test_shape_goldens.py + tests/goldens/regen.py`: both golden suites parametrize over `builtin_names`(), so a builtin orin profile auto-joins the matrix (orin.env + one golden per shape) and regen.py regenerates them — no hand-maintained test lists to chase
  - seeds: `c5`
- `s7` — `lobes/templates/fleet/docker-compose.yml (vllm-multimodal service) + lobes/gateway/_config.py`: the senses lane defaults `MULTIMODAL_MODEL`/`SERVED_NAME` to coolthor and hardcodes the gemma4 parser PAIR (family-wide, checkpoint-agnostic — fine) plus a NON-parameterized --speculative-config literal that tests/`test_catalog.py` byte-guards against the catalog; gateway `_DEFAULT_MULTIMODAL` also names coolthor — these defaults move only on a fleet-wide swap, while a profile-rendered `MULTIMODAL_MODEL` overrides them per-box either way
  - seeds: `c6`, `c11`
- `s8` — `lobes/profiles/schema.py (RoleProfile)`: profiles can express exactly nine knobs (feasible/model/`gpu_mem_util`/`max_model_len`/quantization/`kv_cache_dtype`/`attention_backend`/`enforce_eager`/`max_num_seqs`) — no `speculative_config` knob, so a profile alone cannot change or disable the senses MTP draft today
  - seeds: `c6`
- `s9` — `lobes/profiles/builtin_shapes/thor-lobe.toml`: the shape's \[overrides.senses\] carries Thor-MEASURED values (util 0.30, 131072) that overlay whatever card profile it renders against — on this Orin they clobbered the measured 0.45 and the live deployment hand-edits .env to compensate; shape overrides are per-shape, not per-card
  - seeds: `c7`
- `s10` — `docs/orin-profiles.md`: the live-validated Orin record: senses at util 0.45/131072 (0.30 refused live), KV 802,644 tokens; W4A16-on-Ampere rationale; deliberate absence of a builtin orin profile pre-boot (#108) — now satisfied by this very validation; csv-mode runtime:nvidia hand-edit that re-init reverts; the three assess probes passed on `sm_87`; cortex infeasible = hard architecture line
  - seeds: `c3`, `c4`, `c8`, `c9`, `c10`, `c13`
- `s11` — `~/.lobes deployment dir (.env + profiles/orin.toml, read-only)`: live state today: operator profile orin.toml selected (`LOBES_PROFILE`=orin), `MULTIMODAL_MODEL`=coolthor at util 0.45/131072 with `TRITON_ATTN`, cortex proxied to the Spark via `PRIMARY_PEER_`\* — the swap lands here as a re-render + model change, everything else preserved
  - seeds: `c9`, `c13`
- `s12` — `docs/orin-profiles.md (Shape choice) + CLAUDE.md deployment-shapes section`: this deployment deliberately chose thor-lobe over orin-small (Orin hosts senses, the user's #127 decision); orin-small remains DECLARED/UNVALIDATED and out of this idea's path
  - seeds: `c14`
- `s13` — `user capability statement (2026-08-04) + lobes/catalog.py _SHAPE_GEMMA4_UNIFIED + spark.toml cortex ViT-probe comments`: user wants image+video+audio+reasoning from the QAT checkpoint; repo's own Gemma 4 capability claim is text+image+audio (no video anywhere); audio is blocked by vLLM #101 not the checkpoint; reasoning parser already hardcoded on the lane (muse-validated, 12B-unvalidated); image is the one capability already live-validated on this box (red-image probe, docs/orin-profiles.md)
  - seeds: `c15`, `c16`, `c12`
- `s14` — `challenge pass / counter-evidence lens: HF config.json of unsloth/gemma-4-12B-it-qat-w4a16`: fetched unauthenticated (repo not gated): `max_position_embeddings`=262144, `video_token_id` AND `audio_config` declared, Gemma4UnifiedForConditionalGeneration (existing custom image serves the class), quant = compressed-tensors int4 pack-quantized — NOT the FP4 path this box live-proved; c3 amended accordingly
  - seeds: `c24`, `c3`
- `s15` — `challenge pass / lifecycle + measurement lens: docs/model-switch-playbook.md par.1/4/5/6`: incumbent-first benchmark ordering adopted (c25); the reasoning-vs-`reasoning_content` field trap invalidated h11 as written — rejected and replaced by h20; the par.5 probe designs (reversed-motion video, negative controls) already match h10; par.6 record-refused-utils discipline applies to the c9 live-test
  - seeds: `c25`, `c16`
- `s16` — `challenge pass / adjacent-systems lens: docs/model-switch-playbook.md par.2 + the Spark's senses proxy to this box`: every audited mesh consumer pins the RAW served id (reachy-mini-cli, colleague config, culture.yaml), and the Spark forwards model=senses here — swapping the served name 404s them; the three deliberate options are recorded as q3 for the user
  - seeds: `c11`
- `s17` — `challenge pass / operations lens: ~/.lobes/.env (live read) + orin-box deployment memory`: `LOBES_IOWAIT_DEGRADED_THRESHOLD`=50 persisted on disk while the running gateway holds an ephemeral 100 override — any recreate resurrects the Tegra spurious-iowait 429-shedding of senses; persistence requirement seeded (verify exact knob name in lobes/gateway/`_pressure_policy.py` at implementation)
  - seeds: `c26`
- `s18` — `challenge pass / reversibility lens: lobes/cli/_commands/init.py (_apply_profile_env + scaffold plan)`: .env is merged per-key (operator lines survive a re-render); pre-existing compose files require --force which reverts the csv-mode hand edits (c10's territory); snapshot-based rollback seeded as c27
  - seeds: `c27`, `c10`
- `s19` — `challenge pass / security lens: docs/evidence/2026-07-16-proxy-lobes-live-spark-thor.txt`: the redaction convention exists — keys appear as <...key> placeholders in committed evidence; boundary c28 makes it binding for this swap's transcript
  - seeds: `c28`
- `s20` — `challenge pass / clean lenses: disk capacity, HF gating, .env data-loss`: clean: 1.6T free vs a ~7.7GiB checkpoint (du on the incumbent); config.json fetched without auth so no license gate blocks the pull; .env per-key merge verified in code — residual risk that remains: boot-order memory race on unified boards (parked v7) and everything gated on the live boot itself
  - seeds: `c27`

## Decisions

- MTP policy (user decision 2026-08-04): attempt the existing google/gemma-4-12B-it-assistant draft against the unsloth QAT export at boot; if vLLM refuses it or it fails to load, DROP speculative decoding on the orin senses lane entirely rather than block the swap — which requires the c6 parameterization to support an explicit off/empty value
- context policy (user decision 2026-08-04): the model card is the guide — start `max_model_len` from the card's declared 256K (still reading config.json for the literal `max_position_embeddings` at implementation), and RE-MEASURE live on this box what actually fits; this model + this box may serve MORE than the incumbent 131072

## Open parks

- [unknown_nonblocking] video intake verdict on the unsloth checkpoint: pending the live reversed-motion probe (c15) — unknown until the acceptance run
- [unknown_nonblocking] audio serving verdict on the pinned vLLM nightly: pending the live #101 silent-drop re-probe (c12) — unknown until the acceptance run
- [unknown_nonblocking] the senses swap window is mesh-visible downtime: the Spark forwards model=senses here, so its callers 404/timeout during the reboot; sequence per playbook par.7 and boot sequentially per the Orin boot-ordering caveat — an ops-runbook concern for the plan, not a spec change

## Resolved vagueness

- [unknown_nonblocking] MTP draft compatibility/acceptance: the google/gemma-4-12B-it-assistant draft was trained against the base it-model; coolthor measured 57.9% acceptance — whether it loads against the unsloth QAT export and what acceptance it reaches is unmeasured until the live boot (the speculative-config parameterization in c6 is the off-switch if it fails) — resolved: user decision: try the draft, drop MTP outright if unsupported — never block the swap on it (c17)
- [unknown_nonblocking] native context of the unsloth export: HF card summary says 256K, the repo measured the 12B IT line at 131072 (`text_config`.`max_position_embeddings`) — read the checkpoint's config.json before setting `max_model_len`; the budget hypothesis (c8) assumes 131072 — resolved: user decision: the model card is the guide — start from its declared 256K and re-measure live on this box; config.json still read at implementation (c18)
- [unknown_blocking] video intake on senses: NEITHER the repo's Gemma 4 shape literal (unified text+image+audio, lobes/catalog.py) nor the unsloth HF card mentions video — whether the architecture and vLLM's `gemma4_unified` path serve video at all is unknown; probe live with a motion-reversed negative control (the standard cortex's 2026-07-31 ViT validation set, spark.toml comments) — blocking for the video capability claim only, not for the swap — resolved: user ruling 2026-08-04: downgraded to non-blocking — the c15 probe matrix (reversed-motion control) is the resolution mechanism; the swap ships regardless of the verdict
- [unknown_blocking] audio serving status on the CURRENT pinned vLLM nightly: #101 was measured on the 2026-07 digest — unknown whether a newer digest fixes `gemma4_unified` audio intake; the live-test re-probe arbitrates, and the outcome decides whether the user's audio want is servable this iteration — blocking for the audio capability claim only — resolved: user ruling 2026-08-04: downgraded to non-blocking — the c12 silent-drop-detecting probe on the pinned nightly is the resolution mechanism; the swap ships regardless of the verdict
