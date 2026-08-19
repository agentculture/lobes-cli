# Qwen3.8 cortex rollout notes — read this before you see a 404

**2026-08-19.** The Spark's `cortex` served id changed:

```text
unsloth/Qwen3.6-27B-NVFP4  ->  unsloth/Qwen3.8-27B-NVFP4
```

served now at **1,048,576 tokens (1M) via YaRN**, engine
`vllm/vllm-openai@sha256:8bd082c274fae025b7079498fe1da65182ba1d4c2188c0f5a68c1042c38c3695`
(vLLM `0.26.1rc1.dev942+g5a4c8d992`, official nightly). Live boot evidence:
`docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt`; the standalone engine
spike that isolated checkpoint risk from YaRN risk:
`docs/evidence/2026-08-19-spike-qwen3.8-official-nightly-spark.txt`.

**Why this doc exists.** Per `docs/model-switch-playbook.md` §2, no consumer
in this mesh addresses the fleet by role name — every one of them puts the
**raw served id** on the wire, even the ones that discover via
`/capabilities` first. A served-id swap 404s every one of them, instantly and
silently. This note is addressed to every raw-id pinner, so none of them
learns about the swap via a production 404.

## Who must change, and what

### 1. culture/colleague (mesh agent backend)

Reads the model id from `GET /capabilities` at discovery time, then sends
that **raw id** on every subsequent request — it does not re-resolve per
call. `colleague/colleague/config.py`'s `_DEFAULT_MODEL` (and any cached
discovery state) needs:

```text
unsloth/Qwen3.6-27B-NVFP4  ->  unsloth/Qwen3.8-27B-NVFP4
```

A colleague process that discovered before this swap and never restarts will
keep sending the old id and 404 until it re-discovers or is restarted.

### 2. eidetic

Pins the raw id in `eidetic/memory/embed.py` (embed lane — unaffected by
this swap, which is `cortex`/generate-only) and in whatever module handles
its own generate-lane calls, if any. Audit both; update any occurrence of:

```text
unsloth/Qwen3.6-27B-NVFP4  ->  unsloth/Qwen3.8-27B-NVFP4
```

### 3. reachy-mini-cli

Pins the raw id in `docs/operating-reachy.md`, `reachy/vision/scene.py`, and
`reachy/speech/llm.py` (per the `model-switch-playbook.md` §2 audit). Same
change in each:

```text
unsloth/Qwen3.6-27B-NVFP4  ->  unsloth/Qwen3.8-27B-NVFP4
```

### 4. The lobes agent's own `culture.yaml` (this repo) — ALREADY DONE

`culture.yaml`'s `model:` line was updated by task t6 in this same plan
(`qwen3-8-cortex-upgrade`) and now reads:

```text
model: vllm-local/unsloth/Qwen3.8-27B-NVFP4
```

Nothing further to do here — listed for completeness so this note covers
every raw-id pinner named in the playbook audit.

### 5. Peer `.env` mirrors — any box that PROXIES cortex from this Spark

As of 2026-07-31 the only known proxy direction is **Spark -> others**
(`senses` served from Orin, `worker` served from Thor) — this Spark is a
proxy *source* for cortex, not a sink, for every box currently in the mesh.
But the mesh is mixable, and the rule is general: **any box that declares
`PRIMARY_PEER_ORIGIN` pointing at this Spark** must mirror the new served
name into its own `.env` (`PRIMARY_SERVED_NAME`, and its own vLLM
`--served-model-name` if it runs a passthrough lane) — otherwise its
peer-readiness probe breaks in the specific, silent way recorded on
2026-08-05: the probe checks the peer's own `/v1/models` for the
**advertised** id, so a stale mirror on the peer side shows
`proxied: true` / `ready: false` and the request still 404s, even though
the origin box (this Spark) is healthy and serving the new id correctly.

**Per the operator's Spark-only boundary for this upgrade (c25 in the
acceptance evidence): Thor and Orin deployment state was NOT touched by
this swap.** Their `.env` files still declare the old id where relevant.
Their operators change them when they apply this note — this Spark's own
change does not propagate automatically to any peer.

## Operational notes for every consumer

- **Context is now 1,048,576 (1M) via YaRN**, `gpu_mem_util=0.58` — MEASURED
  at that budget (attempts at 0.60 and 0.58-before-reclaim were both
  REFUSED by vLLM on this unified-memory box; 0.58 booted only after the
  opt-in `embed-deep` gear was stopped to free headroom — see below). KV
  cache measured at 1,271,476 tokens, i.e. a 1.21x ceiling at the full 1M
  depth: this is effectively single-request at that context length, not a
  concurrency win. Don't multiply single-stream throughput by that ceiling
  (the same misreading `CLAUDE.md`'s worker section warns about).
- **The opt-in `embed-deep` lane (`Qwen/Qwen3-Embedding-4B`) is STOPPED on
  this box**, reclaimed to fund the 1M budget (`COMPOSE_PROFILES=embed-deep`
  removed from `.env`; container stopped). Anyone calling `embed-deep` on
  this Spark now gets `role_infeasible` / an absent lane — use the 0.6B
  `embedder` role instead, or re-host `embed-deep` on a different box if the
  4B embedding quality is required.
- **MTP runs with the generic `{"method": "mtp"}` speculative config** —
  the drafter loads from the checkpoint's own MTP module, no external draft
  repo. Measured draft acceptance on the standalone spike was 41.3–47.8%,
  notably lower than the outgoing checkpoint's 61–63%; that gap is an open
  item (t8), not yet explained, and may be generation-content-dependent.
- **The old id now 404s on this gateway.** `unsloth/Qwen3.6-27B-NVFP4` is no
  longer served — it remains a selectable *candidate* via `lobes switch`,
  same as every other catalog entry, but nothing on this box currently
  serves it, and no dual-served-name compatibility shim was used for this
  swap (playbook §2 option 1 was not taken — this was a deliberate,
  coordinated break per option 3, matching the precedent set on 2026-07-31).

## How to verify

```bash
# expect the NEW id, 200:
curl -s http://<spark-gateway>:8001/v1/models | grep -o '"id":"[^"]*"'

# expect 200 (role alias still resolves):
curl -s -o /dev/null -w '%{http_code}\n' \
  http://<spark-gateway>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"cortex","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# expect 404 (old id is retired, not dual-served):
curl -s -o /dev/null -w '%{http_code}\n' \
  http://<spark-gateway>:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"unsloth/Qwen3.6-27B-NVFP4","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'
```

Expect: `/v1/models` lists `unsloth/Qwen3.8-27B-NVFP4` (not the 3.6 id);
`cortex` returns 200; the old raw id returns 404. If any consumer above
still shows the old id in its own logs/config after this note lands, it has
not been updated yet — go fix it there, not here.

## See also

- `docs/model-switch-playbook.md` §2 — the general "know who hardcodes the
  served id" playbook this note follows.
- `docs/qwen3.8-27b-nvfp4.md` — the per-model doc for the new checkpoint
  (config-verified facts, serving knobs).
- `docs/evidence/2026-08-19-accept-qwen38-1m-spark.txt` — the live 1M boot,
  including the 0.60/0.58 refusals and the embed-deep reclaim decision.
- `docs/evidence/2026-08-19-spike-qwen3.8-official-nightly-spark.txt` — the
  standalone engine spike (262144-native leg, MTP acceptance numbers).
