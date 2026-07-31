# thor worker lobe (Qwen3.6-35B-A3B)

> Thor moves off the Gemma 4 31B muse and hosts unsloth/Qwen3.6-35B-A3B-NVFP4 as 'worker' — a new fast ground-work lobe with self-draft MTP
> instruction: ship as a PR chain: catalog entry + eighth-role registry + opt-in gateway/compose/shape wiring + docs, then a live Thor acceptance run (measured budget, MTP acceptance, parser-pair transcript under docs/evidence/); version-bump every PR

## Audience

- Culture-mesh Colleagues that address the fleet by role/tier alias (model=worker via the gateway, lobes capabilities / GET /capabilities), plus the operator running the physical Thor

## Before → After

- Before: Thor serves thor-muse (Gemma 4 31B muse at full 256K, util 0.55, gemma4 parser pair); no worker role exists in the contract; the catalog's only 35B-A3B is the 32K-native mmangkad copy with MTP explicitly not carried
- After: Thor serves unsloth/Qwen3.6-35B-A3B-NVFP4 as 'worker' — the EIGHTH first-class Colleague role, a fast ground-work DOER (repo_action allowed, under cortex direction) with self-draft MTP at 262144 native — hosted via a thor-worker shape; muse is dormant mesh-wide (unhosted, no referral)

## Why it matters

- the mesh gains a fast executor — ~3B-active MoE decode speed with 35B-class knowledge plus MTP self-draft — so ground work stops queueing on the 27B cortex; the 31B muse's 67 GiB on Thor bought creative divergence the mesh uses less than fast execution

## Requirements

- lobes/catalog.py gains a NEW SupportedModel entry for unsloth/Qwen3.6-35B-A3B-NVFP4 with role_hint='worker' — distinct from the existing mmangkad/Qwen3.6-35B-A3B-NVFP4 candidate (32K native, moe_backend=marlin, NO speculative_config: nvidia's MTP config fails on that copy with a weight-shape mismatch). The unsloth export is 262144-token native and ships its OWN MTP module ('can act as its own speculative draft'), card-suggested --speculative-config '{"method": "mtp", "num_speculative_tokens": 2}' — a different method string than the 27B primary's qwen3_5_mtp
  - honesty: the committed catalog entry's native_max_model_len / quantization / speculative_config are read from the unsloth checkpoint's actual config.json + hf_quant_config.json at implementation time, not the card prose — and if the checkpoint disagrees with the card (e.g. MTP fails to load), the entry records what was verified
- the role registry widens to an EIGHTH first-class role: lobes/roles.py needs 'worker' entries in ROLES, ROLE_BACKEND, ROLE_ROLE_HINT, ROLE_PATH (/v1/chat/completions), ROLE_RESPONSIBILITIES, ROLE_FORBIDDEN, and ROLE_MAX_MODEL_LEN_ENV (WORKER_MAX_MODEL_LEN); lobes/profiles/schema.py ROLES (the Profile-machinery set) gains 'worker' so shape overrides can declare its budget knobs
  - honesty: every roles surface (lobes capabilities, GET /capabilities, /v1/models alias resolution) lists exactly eight roles with worker present; no stale seven-role literal survives in code, tests, or docs
- worker becomes the SECOND opt-in core role, mirroring muse end-to-end: shapes.OPT_IN_CORE_ROLES += 'worker'; gateway _config.py wires a WORKER_BASE_URL-gated backend, adds 'worker' to OPT_IN_BACKENDS (infeasible-by-default when unwired — model=worker 404s role_infeasible, never a silent fallback), and adds WORKER_* keys to FEASIBLE_ENV / PEER_ORIGIN_ENV / PEER_PROXY_ENV / PEER_API_KEY_ENV; profiles/builtin/base.toml vetoes worker on unrecognised cards exactly as it vetoes muse
  - honesty: with no WORKER_* env set anywhere, every existing deployment renders byte-identically (goldens prove it); model=worker on a non-hosting box 404s role_infeasible — never a silent fallback
- the fleet compose template gains a profile-gated vllm-worker service + WORKER_* knobs in env.example, and shape_render.OPT_IN_CORE_ACTIVATION_ENV maps worker → COMPOSE_PROFILES gate + WORKER_BASE_URL=<http://vllm-worker:8000>. The worker lane runs on the QWEN-lane image (the vllm-openai nightly pin the primary/minor lanes use), NOT the lobes/vllm-gemma4:local custom image the muse lane needs
  - honesty: the rendered thor-worker deployment boots vllm-worker on the Qwen-lane image on the physical Thor and serves /v1/chat/completions through the gateway
- a new built-in shape (working name thor-worker) hosts ['worker','embedder','reranker','stt','tts'], drops cortex/senses to peers, and carries the FULL worker declaration (model + budget knobs) in its own overrides — the card profiles stay silent on worker, exactly the thor-muse pattern. Its gpu_mem_util / max_model_len MUST be measured on the physical Thor before commit: shape budgets on unified-memory boxes are measured truths, not arithmetic (thor-muse's 0.55 was measured only after the 0.40 hypothesis was refused live)
  - honesty: the shape TOML's committed budget values are the ones a live Thor boot measured, with any refused hypothesis recorded in the TOML comments (the thor-muse 0.40→0.55 pattern)
- the worker lane serves the Qwen-family parser PAIR: --tool-call-parser=qwen3_coder (runtime._parser.infer_parser must resolve the unsloth id to qwen3_coder — the catalog pairing guard in tests/test_catalog.py enforces it) plus --reasoning-parser=qwen3, mirroring the cortex lane — never inferred from the model card, and verified live with skip_special_tokens:false per the per-family parser rule
  - honesty: verified live on the worker lane with skip_special_tokens:false — a tool call parses into a tool_calls array (not content) and the reasoning trace separates via --reasoning-parser=qwen3
- worker's role contract (user decision q1): responsibilities include fast ground-work execution — tool_use + repo_action allowed, under cortex direction; forbidden_responsibilities: final_decision, security_decision. First role besides cortex permitted repo_action
  - instruction: ROLE_RESPONSIBILITIES['worker'] = ground-work execution tokens + tool_use + repo_action; ROLE_FORBIDDEN['worker'] = ('final_decision','security_decision'); update docs/colleague-stack.md division-of-labour table
  - honesty: capabilities output for worker lists repo_action among responsibilities and forbidden is exactly [final_decision, security_decision]; docs/colleague-stack.md's division-of-labour records the widening — every 'cortex is the only lobe that acts' wording is updated

## Honesty conditions

- the announcement is falsifiable against live surfaces: the deployed Thor answers model=worker through the gateway and the acceptance transcript exists under docs/evidence/
- the shape goldens (machine-as-brain / spark / thor / base) render byte-identical before and after the change when no WORKER_* env is set
- the thor-worker shape's hosts list includes embedder/reranker/stt/tts and the rendered deployment serves them unchanged — same probes thor-muse passes today
- after the move, model=muse 404s role_infeasible with no hosted_by on every box; the thor-muse shape, catalog entry, and docs stay in-tree explicitly marked dormant/unhosted
- the surfaces Colleagues actually consume (capabilities JSON keyed by role, gateway tier aliases) carry worker — the audience needs no lobes source access to discover it
- verified live on the deployed Thor: lobes capabilities lists worker feasible+ready serving the unsloth id, and muse feasible:false with no hosted_by
- matches the recorded present: Thor's deployment runs thor-muse per the 2026-07-17 evidence transcript, and the catalog's mmangkad entry says 32K-native with no speculative_config
- the speed rationale is measured, not assumed: the acceptance run records worker-lane decode tok/s with MTP on (and off, if feasible) via lobes measure/benchmark
- every listed signal is a checkable artifact: capabilities JSON, a chat transcript with a parsed tool_calls array, the docs/evidence/ file, and the muse role_infeasible 404 body
- the exported spec and resulting plan contain an explicit live-measurement task on the physical Thor whose outputs (budget, MoE backend, MTP verdict) are committed as shape TOML + catalog data with evidence under docs/evidence/ — no committed value precedes its measurement

## Success signals

- on the deployed Thor: lobes capabilities shows worker feasible+ready with truthful model/context/mtp; model=worker serves a chat completion whose tool call parses into tool_calls (not content); an acceptance transcript under docs/evidence/ records the measured budget + MTP acceptance; model=muse 404s role_infeasible with no hosted_by

## Scope / boundaries

- cortex/senses/embedder/reranker contracts are untouched and machine-as-brain stays byte-identical — worker is opt-in-core like muse: non-hosting shapes render nothing for it, only base.toml's veto emits WORKER_FEASIBLE=false, and a stale pre-worker .env defaults it to infeasible via OPT_IN_BACKENDS
- the pooling gears + audio overlay keep co-residing on Thor (mesh-brain decision 2: cheap gears co-reside on every box that wants them) — the new shape hosts embedder/reranker/stt/tts exactly as thor-muse does today
- muse support is dropped as deployment reality (user decision q3): no box hosts the 31B, Thor's new shape declares no MUSE_PEER_ORIGIN, model=muse 404s role_infeasible without referral. The muse role/gear/shape code stays in-tree as dormant (cite-don't-delete; the q2 tier order still ranks worker < muse), with docs updated to say muse is unhosted
  - instruction: leave muse code/shape/catalog in-tree marked dormant; strip muse hosting from the deployed Thor .env; declare no MUSE_PEER_ORIGIN anywhere; docs say muse is unhosted mesh-wide

## Non-goals

- the mmangkad/Qwen3.6-35B-A3B-NVFP4 catalog entry stays a candidate unchanged (no promotion, no removal); docs/qwen3.6-35b-a3b-nvfp4.md gains the unsloth variant's story rather than being rewritten away

## Assumptions

- the new built-in shape is named thor-worker, mirroring thor-muse's naming (card + hosted-lobe)

## Scope exploration

- `s1` — `lobes/catalog.py + HF card unsloth/Qwen3.6-35B-A3B-NVFP4`: catalog already holds the mmangkad sibling of this checkpoint as a candidate (32K native, marlin verified GB10-solo, MTP explicitly NOT carried — nvidia's config fails on it); the unsloth export verified on HF 2026-07-31: MoE 35B/~3B-active, 262144 native (→1M), includes its own MTP draft module, vLLM card syntax method=mtp num_speculative_tokens=2
  - seeds: `c2`
- `s2` — `lobes/roles.py + lobes/profiles/schema.py`: every role surface is a per-role dict keyed by name — seven roles today ('cortex','senses','muse','embedder','reranker','stt','tts'); muse (added as the seventh) is the exact structural precedent: generate-lane role, _CHAT_PATH, its own backend name, own MAX_MODEL_LEN env. schema.py ROLES is the 5-core subset shapes can override
  - seeds: `c3`
- `s3` — `lobes/profiles/shapes.py + lobes/gateway/_config.py + lobes/profiles/builtin/base.toml`: OPT_IN_CORE_ROLES=('muse',) with machine-as-brain's DEFAULT_HOSTED_ROLES derived by exclusion — adding 'worker' keeps machine-as-brain byte-identical automatically; OPT_IN_BACKENDS=frozenset({'muse'}) implements unwired→infeasible; all four peer-channel env dicts key by backend name; base.toml [roles.muse] feasible=false is the unknown-card veto precedent
  - seeds: `c4`
- `s4` — `lobes/templates/fleet/docker-compose.yml + env.example + lobes/profiles/shape_render.py`: vllm-muse is the template precedent: profile-gated ('muse' compose profile), no host port, MUSE_* knob family, activation env written by the shape render (OPT_IN_CORE_ACTIVATION_ENV={'muse': {'MUSE_BASE_URL': ...}}); muse rides image lobes/vllm-gemma4:local while the Qwen lanes ride vllm/vllm-openai@sha256:7c5a10e9... (VLLM_NIGHTLY_IMAGE) — worker is a Qwen3.6 and belongs on the latter
  - seeds: `c5`
- `s5` — `lobes/profiles/builtin_shapes/thor-muse.toml + tests/goldens/shapes`: thor-muse.toml is the exact structural template: hosts list, [overrides.muse] with model/gpu_mem_util/max_model_len/quantization/attention_backend, header comments recording MEASURED evidence; goldens exist per shape (tests/goldens/shapes + regen.py) so a new shape needs its golden env files regenerated
  - seeds: `c6`
- `s6` — `lobes/runtime/_parser.py + tests/test_catalog.py`: the mmangkad 35B sibling already carries tool_parser='qwen3_coder' in the catalog and the pairing guard asserts catalog↔infer_parser agreement; the gemma4 incident (silent 200-OK failure from a wrong parser pair) is the recorded reason live verification is mandatory
  - seeds: `c7`
- `s7` — `lobes/cli/_commands/up.py + capabilities.py + status.py + measure.py + overview.py + init.py`: the CLI names roles per-verb: up.py maps role→service ('muse'→'vllm-muse') and its help string enumerates all seven roles + colleague-stack; a worker role needs the same per-verb touch (service map, help text, measure family, capabilities rendering) — mechanical, all keyed by the same role name
  - seeds: `c3`
- `s8` — `lobes/gateway/_pressure_policy.py + lobes/catalog.py TIER_ROLE`: tier vocabulary is minor < multimodal < muse < main with muse's role name AS its tier; pressure policy now SHEDS non-minor tiers with 429 under pressure (degrade-to-minor path removed — CLAUDE.md's 'degrades to minor' wording is stale vs the code) and worker needs a declared position in both dicts (they must stay identical, per the in-code comment)
  - seeds: `c3`

## Decisions

- the thor-worker shape ships the thor-muse way: budget knobs (gpu_mem_util / max_model_len), the sm_110 MoE backend choice, and the MTP speculative-config are MEASURED during implementation by a live boot on the physical Thor and committed with the measured evidence in the shape TOML comments; the worker role stays DECLARED/UNVALIDATED (#108) until the acceptance transcript lands under docs/evidence/ — the spec requires the measurement, it does not guess the values
