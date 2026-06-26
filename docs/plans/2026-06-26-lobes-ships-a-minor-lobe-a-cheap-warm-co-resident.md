# Build Plan — lobes ships a 'minor' lobe: a cheap, warm co-resident Qwen3.5-4B small-brain served bf16 alongside the 27B primary, driven by lobes run/route/eval with escalation + confidence governance, and chosen as the first unsloth-LoRA fine-tune target (training itself comes later)

slug: `lobes-ships-a-minor-lobe-a-cheap-warm-co-resident` · status: `exported` · from frame: `lobes-ships-a-minor-lobe-a-cheap-warm-co-resident`

> lobes ships a 'minor' lobe: a cheap, warm co-resident Qwen3.5-4B small-brain served bf16 alongside the 27B primary, driven by lobes run/route/eval with escalation + confidence governance, and chosen as the first unsloth-LoRA fine-tune target (training itself comes later)

## Tasks

### t1 — Catalog: add Qwen3.5-4B 'minor' gear + bf16-unquantized convention

- covers: c5, h9, c14, h3
- acceptance:
  - catalog.py has SupportedModel id='Qwen/Qwen3.5-4B', role_hint='minor', task='generate', shape='hybrid linear-attn + ViT (multimodal)' (no moe_backend), native_max_model_len=262144, tool_parser='qwen3_coder', doc='qwen3.5-4b-minor.md'
  - a quantization sentinel (e.g. 'none') represents bf16; lobes switch/compose OMIT --quantization for it; tests/test_catalog.py passes incl. the new bf16 case; a stub docs/qwen3.5-4b-minor.md exists so test_every_doc_file_exists passes
  - lobes overview --list and GET /v1/models/supported include the minor gear

### t2 — Parser: teach runtime/_parser.py the Qwen3.5 family

- covers: c13, h2
- acceptance:
  - infer_parser('Qwen/Qwen3.5-4B') == 'qwen3_coder'; markers qwen3.5/qwen3-5/qwen3_5 added to the qwen3_coder rule AHEAD of the generic qwen3->hermes rule; every existing id's parser unchanged
  - a unit test pins the value and tests/test_catalog.py::test_tool_parser_matches_infer_parser passes for the new entry

### t3 — Minor client: stdlib-urllib OpenAI client for the minor backend

- covers: c15, h4
- acceptance:
  - lobes/minor/__init__.py + lobes/minor/_client.py post to the gateway /v1/chat/completions using ONLY stdlib urllib (mirrors assess.py); no new runtime deps; importable without optional extras
  - the client is read-only (never writes .env/compose); a unit test exercises it against a stub HTTP server

### t4 — Governance + escalation model for the 'minor' role

- depends on: t3
- covers: c7, h10, c2, h6, c4, h8
- acceptance:
  - lobes/minor/governance.py encodes allowed duties (prepare/classify/format/validate/suggest), forbidden actions (approve/finalize/delete/deploy/architectural), and escalation conditions (needs_codebase_context, security_sensitive, architectural_decision, write_or_delete_operation, final_review_required), keyed by the 'minor' ROLE not a model id
  - a decide(task) function returns handle-vs-escalate; a forbidden action always escalates/refuses; unit tests cover an allowed case, a forbidden case, and each escalation condition

### t5 — Verb: lobes run minor "<prompt>"

- depends on: t3
- covers: c17, c3, h7
- acceptance:
  - lobes/cli/_commands/run.py adds a 'run' verb: lobes run minor "<prompt>" returns a completion from the minor backend via the t3 client; read-only (no --apply); --json emits a structured result
  - a test drives the handler with a stubbed client and asserts stdout + --json; this is the first verb that invokes a model with a prompt (closes the before_state gap)

### t6 — Verb: lobes route "<text>" (across gears + escalate)

- depends on: t3, t4
- covers: c17, h13, c9
- acceptance:
  - lobes/cli/_commands/route.py adds a 'route' verb returning a structured decision {chosen_gear in {minor,primary,candidate}, escalate: bool, confidence: float in [0,1]} via the client + governance; routes ONLY across catalog gears (not tools/agents)
  - a test asserts the decision JSON schema and an escalation case; the confidence source is a defined v1 heuristic

### t7 — Verb: lobes eval minor --suite <path>

- depends on: t3
- covers: c17
- acceptance:
  - lobes/cli/_commands/eval.py adds 'eval': lobes eval minor --suite <path> runs a suite against the minor backend, reporting per-case pass/fail + an aggregate; read-only; missing/empty suite handled gracefully
  - a test runs a tiny fixture suite and asserts the report; suite CONTENTS are out of scope

### t8 — Wire run/route/eval into the CLI

- depends on: t5, t6, t7
- acceptance:
  - lobes/cli/__init__.py imports and registers run/route/eval; lobes --help lists all three and each dispatches to its handler
  - an end-to-end CLI test invokes each verb through main() (argv) with a stubbed client

### t9 — Warm co-residency: fleet compose + gateway routing

- depends on: t1
- covers: c16, h5
- acceptance:
  - templates/fleet/docker-compose.yml gains an OPT-IN vllm-minor generate service (bf16 Qwen/Qwen3.5-4B, --language-model-only, modest VLLM_MAX_MODEL_LEN, bounded gpu-mem) behind the gateway; default behavior unchanged when not opted in
  - the gateway can target minor vs primary for generate traffic (by model id); a test asserts routing + opt-in; GB10 boot/no-OOM is a documented parked assumption, not a CI gate

### t10 — Docs: minor lobe doc, governance, fine-tune target

- depends on: t1, t4, t9
- covers: c8, h11
- acceptance:
  - docs/qwen3.5-4b-minor.md (matching the catalog doc field) documents the gear, allowed/forbidden minor duties + escalation, the read-only/dry-run safety contract, and warm co-residency
  - the doc states bf16 is the FIRST unsloth-LoRA fine-tune target and that fine-tuning is deferred; cosmicproc NVFP4 documented as an untested alternative; markdownlint clean

### t11 — Integration: end-to-end success signals + read-only safety verification

- depends on: t8, t9
- covers: c1, h1, c9, h12
- acceptance:
  - an integration check asserts the observable signals: run returns a completion, route returns decision+confidence, and catalog + parser + 'afi cli doctor . --strict' all pass
  - the read-only contract holds: run/route/eval never mutate .env/compose (verified by doctor / a mutation-safety test)

## Risks

- [unknown_nonblocking] GB10 boot/throughput of the warm co-resident minor (h5) and cosmicproc NVFP4 quality — not load-tested; CI verifies config correctness only (task t9)
- [unknown_nonblocking] confidence/uncertainty metric source (logprob-derived vs self-reported) — v1 uses a defined heuristic; calibration refined later (task t6)
- [follow_up] lobes train / LoRA execution, eval-suite contents, and minor MTP spec-decode are deferred per the spec
