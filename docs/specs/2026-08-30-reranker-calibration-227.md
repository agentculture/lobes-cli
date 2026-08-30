# reranker calibration (#227)

> The reranker lane serves Qwen3-Reranker-0.6B with the model card's judge prompt, so relevance scores are calibrated and a caller's instruction is honored instead of silently dropped
> instruction: Add --chat-template=/usr/local/share/lobes/`qwen3_reranker`.jinja (bind-mounted from a vendored lobes/templates/fleet/`qwen3_reranker`.jinja) to the vllm-rerank command; measure before and after with the #220 probe set through the spark gateway; record in docs/evidence and docs/qwen3-reranker-0.6b.md

## Audience

- Callers of the fleet's /v1/rerank and /v1/score — today eidetic-cli's retrieval lane, tomorrow colleague's #277 retrieval lane — and the lobes operator who decides whether a score is safe to threshold
  - instruction: Name the two callers in the doc's calibration note

## Before → After

- Before: The vllm-rerank lane scores bare query+document concatenations: no --chat-template, so the served build never renders the card's judge prompt (measured 2026-08-30: ~24 prompt tokens per pair), distractors land at 0.5-0.77, one position-independent inversion was observed, and a caller's 'instruction' field is accepted but silently dropped; docs say nothing about thresholds
  - instruction: Captured by the baseline transcript (c5)
- After: The lane renders the model card's judge prompt for every pair (system judge line, <Instruct>/<Query>/<Document>, empty think block), the default instruct is the card's, a per-request 'instruction' actually changes the score, and docs/qwen3-reranker-0.6b.md tells a caller what the scores mean (measured before/after) before they wire a cutoff
  - instruction: Captured by the acceptance transcript (c5) and the doc rewrite (c6)

## Why it matters

- A reranker whose scores are uncalibrated is only safe for top-k ordering; eidetic and colleague's retrieval lanes will want a relevance cutoff eventually, and vLLM's own example shows the lane was simply missing the flag its checkpoint requires — a one-flag defect masquerading as model behaviour
  - instruction: Cite the vLLM example serving line in the spec and the doc

## Requirements

- vllm-rerank adds --chat-template with the Qwen3-Reranker judge prompt (system 'Judge whether the Document meets the requirements...', <Instruct>/<Query>/<Document>, empty <think> block). The served build (vllm 0.26.1rc1.dev942, `io_processor.py` `get_score_prompt`) applies a score template ONLY when one is explicitly passed; with none, Qwen3ForSequenceClassification (no SupportsScoreTemplate) falls to bare query+document concatenation — measured live on spark 2026-08-30: 3 pairs = 71 prompt tokens total, so no judge prompt reaches the model. vLLM's own `qwen3_reranker_online.py` example for this exact --hf-overrides invocation includes the --chat-template flag; lobes never did.
  - instruction: In lobes/templates/fleet/docker-compose.yml vllm-rerank command, add '--chat-template=/usr/local/share/lobes/`qwen3_reranker`.jinja' after the --hf-overrides line; update the one compose golden; lobes init re-render on spark; docker inspect proves the arg
  - honesty: docker inspect of model-gear-vllm-rerank after re-render shows the --chat-template arg, and the vllm boot log shows no 'ChatTemplateResolutionError' fall-through to `default_tokenizer_encode`
- The template ships as a lobes scaffold file (lobes/templates/fleet/`qwen3_reranker`.jinja, a verbatim copy of the image's /vllm-workspace/examples/pooling/score/template/`qwen3_reranker`.jinja) bind-mounted read-only like mg-logwrap.sh and the tool-parser plugin — cite-don't-import, never a path inside the image that a digest bump can move. This adds a third packaged-scaffold file to the deployment lock's deviation-d2 gap (docs/deployment-lock.md) and must be named there.
  - instruction: Copy /vllm-workspace/examples/pooling/score/template/`qwen3_reranker`.jinja out of the pinned image into lobes/templates/fleet/`qwen3_reranker`.jinja; add it to the init scaffold list next to mg-logwrap.sh; bind-mount './`qwen3_reranker`.jinja:/usr/local/share/lobes/`qwen3_reranker`.jinja:ro' on vllm-rerank; name it in docs/deployment-lock.md deviation d2
  - honesty: the vendored jinja is byte-identical to the image's examples/pooling/score/template/`qwen3_reranker`.jinja at the pinned digest (diff recorded in the transcript), and docs/deployment-lock.md deviation d2 names it as a third scaffold file
- Per-request 'instruction' becomes live: vLLM's ScoringRequestMixin folds a top-level 'instruction' into `chat_template_kwargs` and the jinja reads it, defaulting to 'Given a web search query, retrieve relevant passages that answer the query'. Measured live today the field is accepted and silently IGNORED (identical scores, 53 tokens with vs without). docs/qwen3-reranker-0.6b.md documents the default instruct and the per-request override on both /v1/rerank and /v1/score.
  - instruction: After the flag lands, probe /v1/rerank and /v1/score with and without 'instruction' and record the score delta; document the default instruct text and the field on both endpoints in docs/qwen3-reranker-0.6b.md
  - honesty: two /v1/rerank calls differing only in 'instruction' return different `relevance_scores` for at least one pair, on both /v1/rerank and /v1/score
- Before/after live measurement under #108: a transcript in docs/evidence/ re-running the #220 probe set (sky; 'which file lists the ports' -> ledger; 'cats purr' distractor; the NOTICE-vs-toolbatch batching inversion) plus assess.py's France probe, on the untemplated lane first (that baseline is unrecoverable after the change — the model-switch-playbook rule) and then on the templated lane, recording score deltas and whether the inversion persists. Today's untemplated spark baseline: ledger 0.933, cats-purr 0.523, bananas 0.514.
  - instruction: Write scripts or a shell transcript running the #220 probe set + France probe against <http://localhost:8001> on spark; run it BEFORE re-render, save as docs/evidence/<date>-baseline-reranker-untemplated-spark.txt; run again after, save as docs/evidence/<date>-accept-reranker-template-spark.txt
  - honesty: the before-run is executed and saved BEFORE the compose change lands on the box (the untemplated numbers cannot be recovered afterwards)
  - honesty: the transcripts record rerank latency before/after at the doc's 1x5 shape (25 ms warm baseline) — the template roughly triples prefill tokens per pair on a lane sold as cheap, and the doc's benchmark table is updated from the after-run
- docs/qwen3-reranker-0.6b.md records the calibration finding for callers — 'usable for top-k ordering, not thresholds' as measured on the untemplated lane, and whatever the templated re-measurement shows — before anyone wires a cutoff; it also notes the rerank-ordering probe is unverified on the GB10 (#106). The doc's build claim is stale and gets corrected in passing: it cites vllm 0.23.1rc1.dev672 / digest 7c5a10e9, while the compose template pins 8bd082c2 and the live lane reports 0.26.1rc1.dev942.
  - instruction: Rewrite docs/qwen3-reranker-0.6b.md: add a 'Prompt template and calibration' section (judge prompt, default instruct, per-request instruction, measured before/after band, 'top-k ordering, not thresholds' if still true, #106 note); replace the 0.23.1rc1.dev672/7c5a10e9 build line with the compose-pinned digest and the live-reported version
  - honesty: docs/qwen3-reranker-0.6b.md states the measured before/after distractor band and cites the transcript by filename; its build/digest line matches the compose template's pinned digest
- COUNTER-EVIDENCE PROBE (challenge pass, CPU fp32 inside the running container, 2026-08-30): the card's judge template + yes/no logit method scores the reporter's pairs at 1.000 (ledger) / 0.000 (cats purr) / 0.000 (bananas), resolves the NOTICE-vs-toolbatch inversion (0.000 vs 1.000), and the France probe at 0.995/0.000/0.000; bare concatenation on the SAME weights reproduces the served lane almost exactly (0.933/0.533/0.514 vs served 0.933/0.523/0.514; NOTICE 0.786 vs toolbatch 0.988; bananas 0.870 for France). This confirms (a) the served path IS bare concatenation and (b) the template is sufficient to fix both reported symptoms on these pairs. The acceptance transcript must reproduce this on the GPU lane (bf16), not cite the CPU probe.
  - instruction: Re-run the three probe sets (ports/France/toolbatch) against the GPU lane after the change and put the numbers beside the CPU probe's in the acceptance transcript; keep the CPU script under the transcript as an appendix
  - honesty: the GPU acceptance transcript shows the same three probe sets with distractors well below the untemplated values and the inversion resolved; if bf16 on the lane disagrees with the fp32 CPU probe, the transcript says so
- The jinja registers in lobes.runtime.`_compose`.`FLEET_TEMPLATES` ('fleet/`qwen3_reranker`.jinja' -> '`qwen3_reranker`.jinja'), the single registry read by lobes init (`scaffold_plan`/`write_scaffold`), by doctor's `_expected_templates` -> `_scaffold_files_check` (so an absent jinja is a named finding and 'doctor --fix --apply' heals it), and by the lock's file set — NOT a bespoke writer like `write_plugin_file`. This refines c3's instruction.
  - instruction: Add 'fleet/`qwen3_reranker`.jinja': '`qwen3_reranker`.jinja' to `FLEET_TEMPLATES` in lobes/runtime/`_compose.py`; extend the doctor `scaffold_files` test and the init dry-run golden
  - honesty: 'lobes doctor' on a deployment dir with the jinja deleted reports it under `scaffold_files`, and '--fix --apply' restores it byte-identical (unit test)
- Upgrade failure mode is LOUD, and documented as such: a re-rendered compose on a deployment dir that lacks `qwen3_reranker`.jinja makes Docker create a DIRECTORY at the bind-mount source, vLLM's `validate_chat_template`/`load_chat_template` then fails at arg-parse and vllm-rerank crash-loops with /health never green (gateway advertises reranker ready:false). docs/qwen3-reranker-0.6b.md and the CHANGELOG entry name this signature and the fix ('lobes init --apply' or 'lobes doctor --fix --apply' before 'lobes up reranker'); it must never degrade silently to untemplated scoring.
  - instruction: Document the crash-loop signature and recovery in docs/qwen3-reranker-0.6b.md and the CHANGELOG; add a unit test asserting the compose template mounts the file the scaffold writes
  - honesty: the transcript or a unit test shows the crash-loop signature (compose up with the jinja absent) and the documented recovery command
- Observability: lobes assess's rerank probe records usage.`prompt_tokens` (total and per pair) in its result details — the one externally visible signal that the template is rendered — so 'lobes assess --json' on any box shows whether that box is templated. The PASS rule stays ordering-only (c11 unchanged); this adds a detail field, not a gate.
  - instruction: In lobes/assess.py `probe_rerank_correctness`, add usage.`prompt_tokens` (total, `per_pair`) to the details dict; update the probe's unit test fixture
  - honesty: 'lobes assess --json' output on spark after the change carries `prompt_tokens` for the rerank probe and the value matches the transcript

## Honesty conditions

- the live lane's usage.`prompt_tokens` per pair rises to roughly the template's length after the change, proving the prompt is rendered — not just that the flag was accepted
- grep of eidetic-cli and colleague at the time of the change finds no comparison of `relevance_score` against a constant; tests/`test_lobes.py` in colleague still passes
- the gateway's forwarded rerank/score body is byte-identical to the caller's (tests/ pass-through test), before and after
- a diff of the rendered docker-compose.yml before/after shows exactly one added line in vllm-rerank (the --chat-template arg) and one added bind mount; `RERANK_`\* env keys unchanged
- eidetic-cli's embed.py rerank call and colleague's #277 retrieval lane are named as the callers in the doc's calibration note; no other in-mesh /v1/rerank caller exists at grep time
- the before-run transcript records usage.`prompt_tokens` per pair (~24) and the distractor scores on the untemplated lane, so the before-state is evidence, not recollection
- the after-run transcript shows the template's token cost per pair, an instruction-dependent score, and the doc paragraph is quoted in the transcript's 'Deploy record'
- the exported spec cites vLLM's examples/pooling/score/`qwen3_reranker_online.py` serving line as the source of the flag, and the fix is one compose line plus one vendored file — if it needs more, this claim is wrong
- every number in the success signal is read from the transcript, not predicted; if the distractor band does NOT drop, the transcript says so and the doc keeps 'top-k ordering, not thresholds'
- the acceptance probe set includes at least one query with two relevant documents of different quality and the transcript records their scores as distinct, ordered correctly

## Success signals

- docs/evidence/<date>-accept-reranker-template-spark.txt shows the same probe set scored before and after: `prompt_tokens` per pair rises from ~24 to ~85 (template present), the France/Amazon/bananas probe still ranks index 0 first, the 'cats purr' distractor drops materially below the untemplated 0.52, and a request carrying 'instruction' returns a different score than one without
  - instruction: Read every number from the acceptance transcript; record a non-drop honestly

## Scope / boundaries

- The gateway keeps relaying /v1/rerank and /v1/score bodies verbatim (lobes/gateway/server.py POST relay; the rerank/score routes are pass-through with no body rewrite). lobes shapes the instruction inside the engine via --chat-template only — the gateway never injects, rewrites, or defaults an 'instruction' field.
  - instruction: Verify by reading lobes/gateway/server.py's rerank/score relay and the existing pass-through test; make no gateway change
- The lane's other flags are untouched: --runner=pooling --convert=classify, the --hf-overrides (Qwen3ForSequenceClassification / `classifier_from_token` \[no,yes\] / `is_original_qwen3_reranker`), `RERANK_MAX_MODEL_LEN`=8192, `RERANK_GPU_MEM_UTIL`=0.06, and Thor's `sm_110` divergences (`RERANK_ATTENTION_BACKEND`=`TRITON_ATTN`, `RERANK_ENFORCE_EAGER`=--enforce-eager, #105/#106/#109). A --chat-template does not change memory budget (pooling, no decode) but adds ~60 prompt tokens per pair, well inside 8192.
  - instruction: Diff the rendered docker-compose.yml on spark before/after: exactly one added command line and one added volume line under vllm-rerank

## Non-goals

- No checkpoint swap: the catalog entry stays Qwen/Qwen3-Reranker-0.6B loaded via `hf_overrides`; the converted tomaarsen/Qwen3-Reranker-0.6B-seq-cls checkpoint vLLM's examples also mention is not adopted (same template requirement, different weights provenance).
- The assess.py rerank probe fixture and its PASS rule (relevant doc ranks first) are not redefined into a calibration/threshold gate; #106 (probe unverified on GB10) stays its own issue.
- The embedder lane (vllm-embed, Qwen3-Embedding-0.6B) is not touched here even though Qwen3-Embedding has its own instruct-prefix convention; whether it has the same silent-template gap is parked, not fixed, in this frame.

## Assumptions

- The change is a score-VALUE change, not an API break: the only in-mesh consumers are eidetic-cli (eidetic/memory/embed.py maps index->`relevance_score` and sorts; no threshold) and colleague (colleague/lobes.py reads the reranker role and discards it). No caller found pins a cutoff, so no contract migration is needed beyond the doc note; the assess.py France probe (`top_index`==0) is ordering-only and should still pass.
- With the template, scores SATURATE near 0.000/1.000 on easy pairs (fp32 CPU). That makes a threshold safe but may collapse the resolution among several genuinely relevant documents — the opposite failure from today's. The success signal (c17) covers only one-relevant-among-distractors cases; a graded case (two relevant docs of different quality) belongs in the acceptance probe set so 'top-k ordering' is proven to survive the template, not assumed.
  - instruction: Add one graded query (two relevant docs, one clearly better) to the acceptance probe set; record both scores; if they saturate to the same value, the doc says ranking resolution among relevant docs is coarse

## Scope exploration

- `s1` — `lobes/templates/fleet/docker-compose.yml vllm-rerank command (+ live docker inspect of model-gear-vllm-rerank on spark)`: The lane passes --runner=pooling --convert=classify + --hf-overrides and NO --chat-template; the live container's args match the template byte-for-byte, so the gap is in the template, not local drift
  - seeds: `c2`, `c9`
- `s2` — `vllm 0.26.1rc1.dev942 in the served image: entrypoints/pooling/scoring/io_processor.py get_score_prompt + model_executor/models/interfaces.py SupportsScoreTemplate`: A score template is applied only when explicitly provided (FIXME in-source: tokenizer chat templates are deliberately NOT trusted); only `jina_vl` implements SupportsScoreTemplate, so Qwen3ForSequenceClassification takes `default_tokenizer_encode` -> `use_sep_token` defaults True -> tokenizer(text=query, `text_pair`=doc), a bare concatenation with no judge prompt
  - seeds: `c2`
- `s3` — `/vllm-workspace/examples/pooling/score/{qwen3_reranker_online.py,using_template_online.py,template/qwen3_reranker.jinja} in the served image`: vLLM's own serving line for the ORIGINAL checkpoint with these exact `hf_overrides` appends --chat-template examples/pooling/score/template/`qwen3_reranker`.jinja; the jinja is the model card's format verbatim (system judge prompt, <Instruct>/<Query>/<Document>, empty <think> block) with instruction defaulting to the card's 'Given a web search query, retrieve relevant passages that answer the query'
  - seeds: `c2`, `c3`, `c4`
- `s4` — `vllm entrypoints/pooling/scoring/protocol.py ScoringRequestMixin`: Both /v1/rerank and /v1/score accept a top-level 'instruction' that is folded into `chat_template_kwargs`; without a template it is parsed and dropped. Live probe on spark 2026-08-30 confirmed: same scores with and without the field
  - seeds: `c4`
- `s5` — `live /v1/rerank probe through the spark gateway (port 8001), 2026-08-30, read-only`: Untemplated baseline: ledger doc 0.933, 'cats purr' 0.523, bananas 0.514 for the ports query; usage.`prompt_tokens`=71 for three pairs, i.e. ~24 tokens/pair — the ~60-token judge prompt is provably absent. Reporter's 0.72-0.77 distractors were a different query; the distractor band is uncalibrated either way
  - seeds: `c5`, `c6`
- `s6` — `lobes/gateway/server.py (POST relay, lines ~195-215; rerank/score listed as pass-through routes)`: The gateway forwards rerank/score bodies unchanged after the bearer gate; shaping the prompt belongs in the engine flag, not a gateway rewrite
  - seeds: `c8`
- `s7` — `eidetic-cli eidetic/memory/embed.py:300-310 and colleague colleague/lobes.py (reranker role read and discarded)`: eidetic maps index->`relevance_score` and sorts (no threshold); colleague ignores the reranker; no mesh consumer pins a cutoff, so a score-value change is not an API break
  - seeds: `c7`
- `s8` — `lobes/templates/fleet/docker-compose.yml bind mounts (mg-logwrap.sh, vllm_plugins tool parser) + docs/deployment-lock.md deviation d2`: Packaged scaffold files are bind-mounted from the deployment dir; a template jinja follows the same pattern and joins the d2 list of files a lock-only restore is missing
  - seeds: `c3`
- `s9` — `docs/qwen3-reranker-0.6b.md`: Doc claims vllm 0.23.1rc1.dev672 / digest 7c5a10e9; compose pins 8bd082c2 and the live lane reports 0.26.1rc1.dev942 — stale build claim to correct alongside the calibration note; it currently says nothing about thresholds or instruction
  - seeds: `c6`
- `s10` — `lobes/assess.py probe_rerank_correctness + _RERANK_PROBE_* (France/Amazon/bananas)`: PASS is ordering-only (`top_index`==0); it does not read score magnitudes, so it neither detects the calibration gap nor breaks when scores shift
  - seeds: `c7`, `c11`
- `s11` — `tests/goldens (1 golden carries the rerank hf-overrides line; 37 carry RERANK_ keys) + lobes/profiles/render.py RERANK prefix + lobes/runtime/_lock.py allowlist`: A baked flag changes one compose golden; a new `RERANK_CHAT_TEMPLATE` knob would instead touch render.py tables, env.example, 37 goldens and the lock allowlist — the knob-vs-bake choice sets the blast radius
  - seeds: `c9`
- `s12` — `docs/model-switch-playbook.md ordering rule (benchmark the incumbent first)`: The untemplated lane's numbers are unrecoverable once the flag lands; the before-run is mandatory and today's probe is only a partial baseline
  - seeds: `c5`
- `s13` — `lobes/catalog.py Qwen/Qwen3-Reranker-0.6B entry`: Catalog carries `hf_overrides` but has no field for a score/chat template; a template-bearing gear either hardcodes it in compose (as `hf_overrides` already is) or the catalog grows a field — no swap of the entry itself
  - seeds: `c10`
- `s14` — `challenge pass / counter-evidence lens: CPU probe of Qwen3-Reranker-0.6B card method vs concatenation (scratch script, running container, read-only)`: Card method saturates to 0/1 on every reported pair and fixes the inversion; concatenation reproduces the served scores to 2 decimals — the fix hypothesis is now evidence-backed pre-plan, not a guess
  - seeds: `c20`
- `s15` — `challenge pass / failure-mode lens: probe output distribution (0.000/1.000) vs the top-k ordering use case in eidetic`: Saturation is a new, previously unconsidered failure mode; seeded an assumption that the acceptance set must include a graded-relevance case
  - seeds: `c21`
- `s16` — `challenge pass / adjacent-systems lens: lobes/runtime/_compose.py FLEET_TEMPLATES + lobes/cli/_commands/doctor.py _expected_templates/_scaffold_files_check + init.py _apply_fleet_extras`: One registry feeds init, doctor and lock; the plugin file is the exception (bespoke writer) and should not be the model for the jinja
  - seeds: `c22`
- `s17` — `challenge pass / lifecycle lens: wheel upgrade without re-scaffold (docker bind-mount of a missing host path; vllm entrypoints/chat_utils.py validate_chat_template)`: The only silent path would be a template that fails to resolve at request time (ChatTemplateResolutionError -> `default_tokenizer_encode` fall-through); the file-path form fails at boot instead, which is the desired loud failure
  - seeds: `c23`
- `s18` — `challenge pass / overlooked-actors lens: thor-lobe / orin-associate shapes (both host vllm-rerank), Thor's TRITON_ATTN + enforce-eager divergences (#105/#106/#109)`: The template changes prompt length only, not backend or graphs, but nothing on `sm_110`/`sm_87` has scored a templated pair; raised as a rollout-scope question rather than assumed
- `s19` — `challenge pass / adjacent-systems lens: lobes/gateway/_replicas.py fingerprint fields + live GET /capabilities reranker entry (model/runtime/context/ready only)`: No capability field exposes the template; parked as a pool-consistency risk, not in scope for a role whose pools are unvalidated
- `s20` — `challenge pass / observability lens: lobes/assess.py probe_rerank_correctness details dict; live /v1/rerank usage field`: Nothing today distinguishes a templated from an untemplated box from outside; `prompt_tokens` per pair does, cheaply
  - seeds: `c24`
- `s21` — `challenge pass / operations lens: docs/qwen3-reranker-0.6b.md benchmark table (25 ms warm) + lobes/roles_measure.py embed_rerank metrics`: Prefill per pair rises ~3.5x; latency must be re-measured, not assumed negligible
- `s22` — `challenge pass / reversibility lens: compose command line + FLEET_TEMPLATES entry`: Rollback is a one-line compose revert plus re-render; no state migrates and untemplated scores return; the only irreversible thing is the pre-change baseline, already covered by c5/h5
- `s23` — `challenge pass / security lens: HF apply_chat_template sandboxed Jinja; vllm trust_request_chat_template default False; scripts/scan_deployment_secrets.py DEFAULT_SCAN_GLOBS`: The jinja is operator-editable code at the same trust level as compose; callers cannot supply a template; 'instruction' renders verbatim into the prompt at the same trust as query text; the scan globs cover compose/Dockerfiles only, which is fine since a jinja carries no env keys — clean pass
- `s24` — `challenge pass / deployment-lock lens: lobes/runtime/_lock.py [files] digests + docs/deployment-lock.md d2`: The compose digest changes on every box that re-renders, which is expected `lock_drift`; no real box is captured today so nothing drifts in practice; the jinja joins d2's list per c3 — clean pass

## Decisions

- USER DECISION (q1): --chat-template is BAKED into the vllm-rerank compose command, hardcoded like --hf-overrides; no env knob, one compose golden changes
- USER DECISION (q2): the doc's stale vLLM build/digest claim is corrected in this frame alongside the calibration note
- USER DECISION (q3, challenge pass): acceptance is spark-only; thor and orin pick up the template on their next re-render, and the doc names the interim by-box score divergence as accepted

## Hard questions

- Does the jinja's empty <think>\n\n</think> block plus the `from_2_way_softmax` head reproduce the card's reference scores on the card's own example pairs, or does vLLM's seq-cls conversion differ from the card's causal-LM logit read?

## Open parks

- [unknown_nonblocking] Whether the judge template fixes the NOTICE-vs-toolbatch ordering inversion (0.64 vs 0.32) or only the distractor band is unknown until the templated re-measurement; the claim in c5 is a measurement plan, not a predicted outcome
- [unknown_nonblocking] The embedder lane (vllm-embed, Qwen3-Embedding-0.6B) may have the same silent gap for its instruct-prefix convention; unexplored here — candidate follow-up issue
- [unknown_nonblocking] The image's example template path (/vllm-workspace/examples/...) is stable on the pinned digest but unverified across digest bumps; vendoring (c3) sidesteps it, but the vendored copy then needs a drift check against the image on each nightly bump
- [unknown_nonblocking] The replica fingerprint (served id + quantization + max context + runtime; lobes/gateway/`_replicas.py`) does not carry the score template, so a reranker pool mixing a templated and an untemplated replica would rank them 'compatible' while scoring on different scales — the #214 `speculative_config` class of lie. Reranker pools are declared/unvalidated today; if one is ever validated, the template (or a prompt-tokens-per-pair probe) must join the fingerprint. Plan-side risk once the plan exists.
- [unknown_nonblocking] A caller's document plus the ~60-token template may now exceed `RERANK_MAX_MODEL_LEN`=8192 where it previously fit; whether vLLM truncates (`max_tokens_per_doc`) or 400s on that edge is unexamined — the effective max document length shrinks by the template size and the doc should say so
