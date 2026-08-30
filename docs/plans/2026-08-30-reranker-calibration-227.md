# Build Plan — reranker calibration (#227)

slug: `reranker-calibration-227` · status: `exported` · from frame: `reranker-calibration-227`

> The reranker lane serves Qwen3-Reranker-0.6B with the model card's judge prompt, so relevance scores are calibrated and a caller's instruction is honored instead of silently dropped

## Tasks

### t1 — t1 Baseline: probe script + untemplated transcript on spark (BEFORE any compose change)

- instruction: Stdlib urllib only (mirror lobes/assess.py). Read `GATEWAY_API_KEY` from ~/.lobes/.env or .secrets.env if --key is absent. Run on spark from the repo: uv run python scripts/`probe_reranker_calibration.py` --url <http://localhost:8001> | tee docs/evidence/2026-08-30-baseline-reranker-untemplated-spark.txt. Do NOT touch ~/.lobes. Commit the transcript first; nothing else in this wave may re-render the box.
- covers: c14, h11, h5
- acceptance:
  - scripts/`probe_reranker_calibration.py` exists: stdlib-only, takes --url/--key, runs the #220 set (sky; ports→ledger; cats-purr distractor; NOTICE-vs-toolbatch), the France/Amazon/bananas probe, a graded query (two relevant docs, one clearly better), and an instruction/no-instruction pair; prints per-doc scores, usage.`prompt_tokens` total and per pair, and 1x5 rerank latency (median of 5)
  - docs/evidence/2026-08-30-baseline-reranker-untemplated-spark.txt is the script's verbatim output against <http://localhost:8001> on spark, with a header recording git sha, docker inspect of model-gear-vllm-rerank (no --chat-template), and the vLLM version
  - the transcript shows ~24 `prompt_tokens` per pair and identical scores with and without instruction

### t2 — t2 Vendor `qwen3_reranker`.jinja + register in `FLEET_TEMPLATES` (init/doctor/lock)

- instruction: Extract the jinja with: docker run --rm --entrypoint cat vllm/vllm-openai@sha256:<pinned digest from docker-compose.yml> /vllm-workspace/examples/pooling/score/template/`qwen3_reranker`.jinja > lobes/templates/fleet/`qwen3_reranker`.jinja (or docker cp from the running container). Add the `FLEET_TEMPLATES` entry next to `LOG_WRAPPER` with a comment citing #227 and the vLLM example. Update tests/goldens for init dry-run; add the doctor `scaffold_files` test beside the existing plugin-file test. Package data: confirm pyproject includes lobes/templates/\*\* so the wheel ships it.
- covers: c3, c22, h18
- acceptance:
  - lobes/templates/fleet/`qwen3_reranker`.jinja is byte-identical to /vllm-workspace/examples/pooling/score/template/`qwen3_reranker`.jinja in the pinned image (sha256 recorded in the file's sibling comment in `_compose.py` or the PR description)
  - `_compose`.`FLEET_TEMPLATES` maps 'fleet/`qwen3_reranker`.jinja' -> '`qwen3_reranker`.jinja'; lobes init dry-run golden lists it; lobes init --apply writes it
  - test: lobes doctor on a deployment dir with the jinja deleted reports it under `scaffold_files`, and doctor --fix --apply restores it byte-identical
  - uv run pytest -n auto passes; black/isort/flake8 clean

### t3 — t3 Compose: --chat-template arg + read-only bind mount on vllm-rerank; goldens; mount-matches-scaffold test

- instruction: Model the mount on the mg-logwrap.sh line of vllm-rerank. Keep the compose comment style: cite #227, the vLLM example serving line, and why the path form (not literal) — it fails loud at arg-parse when the file is absent. The mount-matches-scaffold test parses the compose YAML, collects volume sources starting with ./ that have an extension, and asserts each is in `FLEET_TEMPLATES` values or the plugin dest name.
- depends on: t2
- covers: c2, c9, c23
- acceptance:
  - lobes/templates/fleet/docker-compose.yml vllm-rerank gains exactly one command line '--chat-template=/usr/local/share/lobes/`qwen3_reranker`.jinja' after --hf-overrides and one volume line './`qwen3_reranker`.jinja:/usr/local/share/lobes/`qwen3_reranker`.jinja:ro'; no other service or `RERANK_`\* key changes
  - the one compose golden carrying the rerank hf-overrides line is updated and every golden test passes
  - test: every bind-mount source under ./ in the fleet compose that is not a directory is a name `FLEET_TEMPLATES` (or the plugin writer) materialises — so a scaffold can never omit a file the compose mounts

### t4 — t4 assess: rerank probe reports usage.`prompt_tokens` (total, `per_pair`) in details

- instruction: Details dict gains `prompt_tokens`: {'total': usage.`prompt_tokens`, '`per_pair`': round(total/len(docs), 1)} or None when usage is absent. Extend tests/`test_assess`\*.py with the fake-response fixture; keep `_RERANK_PROBE_`\* untouched.
- covers: c24
- acceptance:
  - `probe_rerank_correctness` details carry `prompt_tokens`={total, `per_pair`} read from the response's usage; PASS rule unchanged (`top_index` == expected)
  - unit test with a fake response asserts the fields; a response without usage yields `prompt_tokens`=None, not a crash

### t5 — t5 Gateway pass-through: prove rerank/score bodies are relayed byte-identical

- instruction: Look for an existing pass-through/relay test in tests/`test_gateway_`\*.py first (grep 'rerank'). Add the /v1/score + 'instruction' cases to it; assert the fake backend received byte-identical body. No production code change.
- covers: c8, h8
- acceptance:
  - a test posts a /v1/rerank and a /v1/score body (including an 'instruction' key) through the gateway to a fake backend and asserts the backend received the exact bytes; if such a test already exists, cite it in the PR and add only the 'instruction' case
  - no change to lobes/gateway/server.py

### t6 — t6 Acceptance on spark: re-render, GPU measurement, acceptance transcript

- instruction: Operator task on spark, sequenced: (1) verify t1's transcript is committed; (2) from the merged branch: uv run lobes init --apply --force (dry-run first, keep the printed diff); (3) diff ~/.lobes/docker-compose.yml against the pre-render copy; (4) uv run lobes up reranker --apply; watch lobes logs reranker for template errors; (5) re-run the t1 script to the accept transcript; (6) uv run lobes assess --json --role reranker. Header must include git sha, digest, vLLM version, docker inspect args. If bf16 disagrees with the CPU probe, write that down — do not tune.
- depends on: t1, t2, t3, t4
- covers: c1, h1, h2, h3, c4, h4, c5, h15, c15, h12, c17, h14, c20, h16, h20, h19, h9
- acceptance:
  - lobes init --apply --force re-renders ~/.lobes from the branch's source; diff of docker-compose.yml before/after shows exactly one added command line and one added volume line under vllm-rerank; `qwen3_reranker`.jinja present and byte-identical to the image's
  - docker inspect shows the --chat-template arg; the rerank boot log has no ChatTemplateResolutionError; /health green; GET /capabilities reranker ready:true
  - docs/evidence/<date>-accept-reranker-template-spark.txt = the t1 script's verbatim output after the change, with the same header, plus the t1 baseline numbers side by side and the CPU card-probe numbers as an appendix
  - measured: `prompt_tokens` per pair ~85; France probe index 0 first; cats-purr below the baseline 0.52; NOTICE-vs-toolbatch ordering recorded (resolved or not); graded pair scores distinct and correctly ordered (or saturation recorded honestly); instruction vs no-instruction scores differ; 1x5 latency before/after recorded
  - lobes assess --json on spark shows `prompt_tokens` for the rerank probe matching the transcript

### t7 — t7 Docs: qwen3-reranker-0.6b.md calibration section, stale build line, crash-loop recovery, deployment-lock d2, CHANGELOG

- instruction: Read the accept transcript before writing a single number. Section order: What it is / Serving (add the flag + mount) / Prompt template and calibration (new) / API shapes (add instruction) / Upgrade note (crash-loop + recovery) / Benchmark (after-run latency). Update CLAUDE.md's reranker pointer and docs/deployment-lock.md d2 list. markdownlint-cli2 --fix.
- depends on: t6
- covers: c6, h6, c13, h10, c16, h13
- acceptance:
  - docs/qwen3-reranker-0.6b.md gains 'Prompt template and calibration': the judge prompt, the default instruct text, the per-request instruction field on /v1/rerank and /v1/score, the measured before/after distractor band and graded-pair result citing both transcripts by filename, a threshold guidance sentence that follows the numbers ('top-k ordering, not thresholds' kept only if still true), the #106 note, the effective max document length (8192 minus the template), and the spark-only rollout with the accepted thor/orin divergence window
  - the doc's build line names the compose-pinned digest and the live-reported vLLM version; the benchmark table carries the after-run latency
  - the doc and CHANGELOG name the missing-jinja crash-loop signature and the recovery (lobes init --apply or lobes doctor --fix --apply before lobes up reranker); eidetic-cli and colleague #277 are named as the callers; vLLM's examples/pooling/score/`qwen3_reranker_online.py` serving line is cited
  - docs/deployment-lock.md deviation d2 lists `qwen3_reranker`.jinja as the third scaffold file; CLAUDE.md's reranker sentence points at the new section; markdownlint-cli2 clean

### t8 — t8 Version bump (minor) + PR

- instruction: python3 .claude/skills/version-bump/scripts/bump.py minor reads a changelog JSON on stdin — pipe it, then uv lock and commit the re-pin. Open the PR with the cicd skill; sign '- lobes (Claude)'. PR body: validated on spark only; thor/orin declared; link both transcripts, #227, #220.
- depends on: t5, t7
- acceptance:
  - python3 .claude/skills/version-bump/scripts/bump.py minor run with a changelog JSON on stdin; uv.lock re-pinned and committed; version-check, secrets-scan, afi rubric and the full CI matrix green
  - PR body links #227 and #220, both evidence transcripts, and states what is validated (spark) vs declared (thor/orin)

## Risks

- [unknown_nonblocking] The CPU probe used the card's causal-LM yes/no logit read; vLLM's `from_2_way_softmax` seq-cls head on the GPU (bf16) may not reproduce it — t6's transcript is the arbiter and records disagreement rather than hiding it (task t6)
- [unknown_nonblocking] t1 MUST land on the box before any compose change touches ~/.lobes on spark: the untemplated numbers are unrecoverable. Workforce ordering: t1's transcript is committed before t6 begins; t2-t5 never run lobes init --apply on spark (task t1)
- [unknown_nonblocking] t6 restarts a serving lane on spark (lobes up reranker --apply after re-render): eidetic's rerank calls fail for the boot window; operator gate, run outside a mesh-busy period (task t6)
- [follow_up] The replica fingerprint does not carry the score template; a reranker pool mixing templated and untemplated boxes would rank them compatible on different score scales (#214 class). Reranker pools are declared/unvalidated; if one is validated, add the template (or `prompt_tokens`-per-pair) to the fingerprint
- [follow_up] thor and orin serve untemplated scores until their next lobes init --apply (decision q3): open a follow-up issue to re-render both and probe on `sm_110`/`sm_87`, where nothing has scored a templated pair
- [follow_up] vllm-embed (Qwen3-Embedding-0.6B) may have the same silent instruct-prefix gap; unexplored — follow-up issue
- [unknown_nonblocking] A document that previously fit 8192 may now exceed it by the template's ~60 tokens; whether vLLM truncates or 400s is unexamined — t7 documents the shrunken effective length, the edge itself is not tested (task t7)
