# Build Plan — lobes preserves Qwen thinking traces across multi-turn agent loops on the vLLM cortex, so historical <think> reasoning is carried back into context instead of discarded

slug: `lobes-preserves-qwen-thinking-traces-across-multi` · status: `exported` · from frame: `lobes-preserves-qwen-thinking-traces-across-multi`

> lobes preserves Qwen thinking traces across multi-turn agent loops on the vLLM cortex, so historical <think> reasoning is carried back into context instead of discarded

## Tasks

### t1 — Add --default-chat-template-kwargs preserve_thinking to both compose templates

- covers: c3, c4, c8, h1, h7
- acceptance:
  - both lobes/templates/docker-compose.yml and lobes/templates/fleet/docker-compose.yml carry --default-chat-template-kwargs '{"preserve_thinking": true}' on the vLLM primary/cortex service, adjacent to the existing --reasoning-parser=qwen3
  - a test parses both compose YAMLs and asserts the arg is present on the primary/cortex service ONLY (not on the embed/rerank pooling services, which omit reasoning flags)

### t2 — Make the minor generate client reasoning-aware: capture and re-send assistant reasoning across turns

- covers: c9, h6, h9
- acceptance:
  - lobes/minor/_client.py can accept prior conversation turns and, given a response whose assistant message has a populated 'reasoning' (or 'reasoning_content'), builds the next assistant history message preserving that field alongside 'content'
  - a urllib-mocking unit test asserts (a) a two-turn call includes the prior assistant 'reasoning' in the outbound messages, and (b) single-turn calls and the enable_thinking=false path are unchanged when no reasoning is present — no other lane touched

### t3 — Add a read-only two-turn token-delta diagnostic proving preserve_thinking is live

- covers: c7, c10, h2, h3, h4, h10
- acceptance:
  - a new function in lobes/assess.py runs two two-turn probes against the served model (one re-sending the assistant 'reasoning' in history, one content-only) and returns each request's usage.prompt_tokens plus the delta
  - a read-only CLI surface (new verb registered in lobes/cli/__init__.py, or a lobes assess flag) prints both prompt-token counts and the delta, exits 0, and never mutates the deployment
  - a test mocking the server (prompt_tokens rises when <think>/reasoning is present in history) asserts the reported delta is positive with reasoning preserved and ~0 without

### t4 — Document preserve_thinking on the cortex lane (CLAUDE.md + Qwen3.6 model doc)

- covers: c1, c2, c5, c6, h5, h8
- acceptance:
  - CLAUDE.md notes preserve_thinking=true is default-on for the cortex/main lane (per-request overridable), route stays enable_thinking=false, and pooling/senses lanes are untouched
  - docs/qwen3.6-27b-text-nvfp4-mtp.md documents the --default-chat-template-kwargs flag and the two-turn diagnostic; markdownlint-cli2 passes on the changed docs

### t5 — Integration: version bump, CHANGELOG entry, full green suite

- depends on: t1, t2, t3, t4
- acceptance:
  - version-bump minor applied (pyproject.toml) with a CHANGELOG entry citing #93; uv run pytest -n auto passes; black/isort/flake8/bandit clean
