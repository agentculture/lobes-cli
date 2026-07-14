# Build Plan — lobes serves strict, grammar-constrained tool calls with thinking enabled: the cortex ships a think-aware tool-parser plugin so structural-tag decoding survives </think>, and every OpenAI tools caller gets well-formed tool calls instead of parser-salvage mangles — Colleague #320's read_file" + empty-arguments failure is impossible by construction

slug: `lobes-serves-strict-grammar-constrained-tool-calls` · status: `exported` · from frame: `lobes-serves-strict-grammar-constrained-tool-calls`

> lobes serves strict, grammar-constrained tool calls with thinking enabled: the cortex ships a think-aware tool-parser plugin so structural-tag decoding survives </think>, and every OpenAI tools caller gets well-formed tool calls instead of parser-salvage mangles — Colleague #320's read_file" + empty-arguments failure is impossible by construction

## Tasks

### t1 — Think-aware vLLM tool-parser plugin module in the lobes package: lobes/vllm_plugins/{__init__,_thinking,qwen3_thinking_tool_parser}.py + tests/test_vllm_plugin_thinking.py. Pure helper effective_reasoning(request) decides thinking from chat_template_kwargs.enable_thinking (absent→True); plugin subclasses the image's Qwen3EngineToolParser, overrides get_structural_tag to pass that decision, registers as qwen3_coder_thinking, and asserts its import surface loudly at load. vllm imports are guarded so the module is importable (and the helper fully testable) in CI where vllm is absent. CONTRACT PIN: parser name 'qwen3_coder_thinking', plugin path lobes/vllm_plugins/qwen3_thinking_tool_parser.py

- covers: c6, c7, h2
- acceptance:
  - effective_reasoning: absent chat_template_kwargs→True, enable_thinking=false→False, =true→True — unit-tested in CI with no vllm installed
  - with a stubbed vllm surface, the plugin registers under 'qwen3_coder_thinking' and passes reasoning=<helper result> to the structural-tag registry call (test asserts the forwarded kwarg)
  - with a mismatched/missing Qwen3EngineToolParser or changed get_structural_tag signature, plugin load raises a loud, named error (test via stub); module import without vllm does not raise

### t2 — Fleet template + init wiring: lobes/templates/fleet/docker-compose.yml mounts the scaffolded plugin file into vllm-primary and adds --tool-parser-plugin=<mounted path>; PRIMARY_TOOL_CALL_PARSER default flips qwen3_coder→qwen3_coder_thinking; lobes init materialises the plugin file into the deployment dir (like _readiness.py/listen_server.py); env.example documents PRIMARY_TOOL_CALL_PARSER's new default AND the GATEWAY_FORCE_STRICT_TOOLS knob line (all env.example edits live in THIS task for file-disjointness); tests/goldens regenerated. Cortex/main generate lane only

- covers: c6, c8, h4
- acceptance:
  - rendered fleet compose: vllm-primary gains the volume mount + --tool-parser-plugin arg + qwen3_coder_thinking default; goldens updated and passing
  - senses/embed/rerank/minor service definitions are byte-identical before/after (template diff test)
  - lobes init (dry-run + apply in a temp dir) lists and writes the plugin file next to the compose; legacy single-model template untouched

### t3 — Gateway force-strict knob: lobes/gateway/ injects function.strict=true on cortex-lane (main/hard/cortex alias) chat-completions requests that carry tools when GATEWAY_FORCE_STRICT_TOOLS is truthy; on an upstream schema-compile failure it retries ONCE without strict and logs the offending schema; knob-off is byte-identical passthrough. Tests in tests/test_gateway_* with a stubbed upstream. CONTRACT PIN: env knob name GATEWAY_FORCE_STRICT_TOOLS; no env.example edits here (owned by the template task)

- covers: c9, h3
- acceptance:
  - knob unset/0: forwarded request bytes identical to today for tools and tool-less requests alike (byte-compare test)
  - knob on: only cortex-lane chat requests WITH tools gain strict:true on every function tool; senses/minor/embed/rerank lanes and tool-less requests unchanged (tests)
  - stubbed upstream returning a grammar/schema-compile error triggers exactly one retry without strict, returns the retry's response, and logs the offending schema; a non-compile error does NOT trigger the retry (tests)

### t4 — Docs: CLAUDE.md model/tool-calling note, docs/qwen3.6-27b-text-nvfp4-mtp.md strict-tool-calling section (root cause, the reasoning=False upstream gap, the plugin), docs/openai-api.md + docs/gateway-fleet.md gateway-knob + retry-fallback documentation, and the boundary (no image rebuild, no colleague edits, upstream issue as follow-up)

- depends on: t1, t2, t3
- covers: c10
- acceptance:
  - docs state the plugin mechanism, the knob (default off), the retry-without-strict fallback, and the pinned-image boundary; doc-test alignment check passes
  - markdownlint-cli2 clean on every touched doc

### t5 — Live verification on the Spark GB10 (main-agent, on-box): re-deploy the fleet with the plugin + knob and run the four-way #320 replay matrix from the captured request bytes — (1) baseline non-strict still answers, (2) strict+thinking returns clean structured read_file with schema-valid args (today 500), (3) strict+enable_thinking=false stays clean, (4) real 'colleague work' item on a throwaway repo delivers changed files (today write-no-changes); plus lobes assess probes, MTP SpecDecoding metrics present, and a PR-diff surface check (lobes repo only, image digest unchanged)

- depends on: t1, t2, t3
- covers: c1, h8, c2, h9, c3, h10, c4, h7, c5, h11, c11, h5, h1, h12
- acceptance:
  - all four matrix legs pass against the live re-deployed fleet using the captured request bytes (replay-base/replay-strict payloads), transcript archived under docs/evidence/
  - lobes assess correctness probes pass; SpecDecoding metrics present in the primary log during the strict legs (MTP active)
  - the colleague work item completes a multi-turn tool loop with changed files (multi-turn </think> grammar satisfiability proven end-to-end)
  - PR diff touches only the lobes repo; vLLM image digest unchanged

### t6 — Comms (main-agent): reply on colleague#320 with the proven root cause + fix, recommending Colleague also send strict:true client-side (user decision c14); file the reasoning=False hardcode upstream on vllm-project/vllm with the minimal repro

- depends on: t5
- acceptance:
  - colleague#320 reply posted, signed per convention, linking the lobes spec/PR and the evidence transcript
  - upstream vLLM issue filed with the grammar-rejected-token repro and the _apply_structural_tag(reasoning=False) call-site reference; link recorded in the frame's parked follow-up

## Risks

- [unknown_nonblocking] Exact detection signature of an xgrammar schema-compile failure at the gateway (status code + body shape) is unverified — discover it live during the gateway task / verification; the retry heuristic must not swallow unrelated 4xx/5xx (task t3)
- [unknown_nonblocking] ToolParserManager.import_tool_parser file-path semantics in the pinned image (module path vs file path, import timing before serving) verified only by reading source — first live boot in the acceptance run is the proof (task t2)
- [follow_up] Knob default-on policy and upstream vLLM fix tracking are parked follow-ups (frame v1/v2) — not blocking this plan
