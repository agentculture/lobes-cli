# Lightning worker rollout notes — read this before you see a 404

**Not yet live.** This note is written **ahead of** the flip (thor-worker-lobe
plan, tasks t1–t9): the Thor `worker` lane still serves
`unsloth/Qwen3.6-35B-A3B-NVFP4` today. Nothing in this document describes a
completed swap — it is the pre-flip warning the playbook (below) says every
raw-id pinner needs **before** it happens, not after.

The Thor `worker` served id is changing:

```text
unsloth/Qwen3.6-35B-A3B-NVFP4  ->  nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
```

**Why this doc exists.** Per `docs/model-switch-playbook.md` §2, no consumer
in this mesh addresses the fleet by role name — every one of them puts the
**raw served id** on the wire, even the ones that discover via
`/capabilities` first. A served-id swap 404s every one of them, instantly and
silently. This note is addressed to every raw-id pinner found by the audit
below, so none of them learns about the swap via a production 404.

**The contract also changes, not just the id.** The outgoing
`unsloth/Qwen3.6-35B-A3B-NVFP4` is **multimodal** (image+video intake) and a
general-purpose MoE. The incoming Lightning checkpoint is **text-only**
(hybrid Mamba-2 + sparse-MoE, ~3B active) and is explicitly **non-coding** —
action selection, tool loops, RAG, digestion, repo inspection, never code
authoring or final judgment. See issue #187 for the full responsibility
split (`lobes/roles.py` `ROLE_RESPONSIBILITIES['worker']`, task t4 of this
plan). Any consumer that sent worker an image, a video clip, or a
code-authoring task will need to stop — that traffic has nowhere to land on
the new checkpoint, id swap or not.

## Audit method

Per the task brief, no sibling checkouts exist on this box, so the audit used
GitHub code search (`gh api search/code`) against the `agentculture` org,
plus an in-repo grep of this checkout. Every query and its raw hit count is
recorded verbatim in the [Appendix](#appendix-audit-queries-and-raw-results)
so a reviewer can re-run them — absence of a hit is evidenced, not assumed.

## Who must change, and what

### In-repo (this checkout) — owned by other tasks in this plan, not this note

The old id appears in dozens of in-repo surfaces: `lobes/catalog.py`,
`lobes/gateway/_config.py` (`_DEFAULT_WORKER`), `lobes/roles.py`,
`lobes/profiles/builtin_shapes/thor-worker.toml`, `lobes/profiles/builtin/thor.toml`,
`lobes/templates/fleet/env.example` and `docker-compose.yml`, every
`tests/goldens/shapes/thor-worker__*.env` golden, `tests/test_catalog.py`,
`tests/test_roles.py`, `tests/test_worker_compose.py`,
`tests/test_gateway_proxy.py`, `tests/test_gateway_config_wiring.py`,
`tests/test_shape_contract_matrix.py`, `tests/test_parser.py`,
`tests/test_catalog_tiers.py`, `CLAUDE.md`, `README.md`,
`docs/qwen3.6-35b-a3b-nvfp4.md`, `docs/deployment-shapes.md`,
`docs/gateway-fleet.md`, `docs/colleague-stack.md`,
`docs/mistral-small-3.2-24b-nvfp4.md`, `docs/gemma-4-31b-nvfp4.md`,
`docs/qwen3.6-27b-nvfp4-multimodal.md`, `docs/tuning-profiles.md`,
`lobes/cli/_commands/up.py`, `lobes/explain/catalog.py`.

**These are in scope for tasks t3 (catalog demotion), t9 (shape/env/compose
commit), and t15 (docs follow the shipped state) elsewhere in this plan —
not this note.** Listed here only so the audit is complete; t3/t9/t15 own the
actual edits. Historical evidence transcripts
(`docs/evidence/2026-07-31-accept-worker-thor.txt`,
`docs/evidence/2026-07-31-accept-worker-proxy-spark.txt`) and closed
plan/spec/frame records under `docs/plans/`, `docs/specs/`, `.devague/`, and
`.eidetic/memory/` are **historical record — never edited for a swap**; they
describe what was true when they were written.

### External consumers — audited, no live raw-id pin found

The external audit (queries + hit counts in the Appendix) turned up **no
external repo that hardcodes the raw worker id as a live default or
config value**. Specifically:

- **`agentculture/colleague`** — `colleague/oilcheck/three_tier.py` probes
  the worker seat by **discovering** `worker_role.model` from the lobes
  gateway's own role advertisement (`_worker_model_match`, comparing the
  discovered id against `/v1/models` live) rather than hardcoding a served
  id. `colleague/colleague/config.py`'s `_DEFAULT_MODEL` — the field that
  *did* break on the cortex swap (`docs/qwen38-rollout-notes.md` item 1) —
  has no worker-specific counterpart; three-tier/worker dispatch is
  discovery-only. `colleague/oilcheck/tool_calling.py` mentions the raw id
  only in a benchmark-comment table (measured token counts), not as a pinned
  value. **No file to change here for the id swap itself** — but re-verify
  after the flip: `_worker_model_match` will (correctly) start reporting a
  mismatch until Thor's `/v1/models` advertises the new id, which is the
  intended, self-healing behaviour, not a bug.
- **`agentculture/reachy-mini-cli`** — the embodiment layer already
  addresses worker by **role name**, not raw id:
  `REACHY_EMBODY_WORKER_MODEL` defaults to the literal string `worker` (the
  lobes role alias), resolved by the gateway itself
  (`docs/operating-reachy.md` row for `REACHY_EMBODY_WORKER_MODEL`; also see
  `docs/specs/2026-08-01-embodiment-layer.md`, which spells out today's raw
  id only in explanatory prose ("worker = unsloth/Qwen3.6-35B-A3B-NVFP4 on
  thor"), never as a literal pinned in code). This is playbook §2's option 2
  (migrate to role names) — already done for worker, unlike cortex/senses.
  **No code change needed here.** The prose mentions of the current id in
  `docs/specs/2026-08-01-embodiment-layer.md` will read as stale after the
  flip; that is a reachy-mini-cli docs follow-up, not a functional break.
- **`agentculture/embodiment`** — `README.md` and
  `docs/live-test-results/worker-toolloop.md` cite the raw id in a **closed
  experiment report** ("role `worker` = `unsloth/Qwen3.6-35B-A3B-NVFP4`
  (proxied)", 2026-08-04 measurement). This is a historical record of what
  was tested, not a live consumer that sends the id on the wire — it will
  read as an accurate description of *that* run forever, and does not need
  editing for the swap to be safe. No functional break here.
- **`agentculture/eidetic-cli`**, **`agentculture/steward`**,
  **`agentculture/guildmaster`** — clean; see Appendix.

**Net finding: the worker swap has NO known external raw-id pinner that will
404.** This differs from the cortex swap
(`docs/qwen38-rollout-notes.md`), where colleague, eidetic, and
reachy-mini-cli all needed code changes. The likely reason: worker is a
newer, opt-in role (introduced 2026-07-31) and its main external consumers
were built discovery-first from the start. Treat this as a snapshot, not a
guarantee — GitHub code search can miss matches (private repos, code search
indexing lag, forks); re-run the Appendix queries closer to the actual flip.

### Peer `.env` mirrors — any box that PROXIES worker from Thor

Per the 2026-08-05 cortex lesson
(`docs/qwen38-rollout-notes.md` §5, and repeated verbatim in this plan's own
spec/frame): **any box that declares `WORKER_PEER_ORIGIN` pointing at Thor
must mirror the new `WORKER_SERVED_NAME` into its own `.env`** the moment
Thor's worker lane flips — otherwise its peer-readiness probe breaks in the
exact silent way already recorded once. The probe checks the peer's own
`GET /v1/models` for the **advertised** id; a stale mirror on the proxying
box shows `proxied=true` / `ready=false`, and every `model=worker` request
through that box still 404s, even though Thor itself is healthy and serving
the new id correctly.

Known `WORKER_PEER_ORIGIN` declarations today (grep of this repo's own
in-tree records, not live `.env` state — those live only on the boxes
themselves): the DGX Spark and the Orin have both been recorded declaring
`WORKER_PEER_ORIGIN` pointing at Thor's gateway
(`.devague/frames/every-lobe-in-the-mesh-can-see-the-spark-reaches-t.json`,
`.devague/frames/unsloth-qat-senses-first-class-orin-variation.json`).
**Both boxes' `.env` files must add or update, on the day Thor's worker lane
flips:**

```bash
WORKER_SERVED_NAME=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
# WORKER_PEER_ORIGIN and WORKER_PEER_PROXY stay unchanged — only the
# advertised served name needs to move.
```

This Thor-side note does not, by itself, touch the Spark's or the Orin's
deployment state — that is each operator's own follow-up, the same boundary
the qwen38 note drew for cortex ("this Spark's own change does not
propagate automatically to any peer").

## Operational notes

- **The swap has not happened yet.** `unsloth/Qwen3.6-35B-A3B-NVFP4` is still
  the served worker id as of this writing. This note exists so the mirror
  step (above) can happen the same day as the flip, not discovered after a
  wave of 404s.
- **Contract narrows, not just the id**: text-only (no more
  image/video intake on worker), non-coding (repo inspection and tool loops
  stay in scope; code authoring and final judgment do not) — see issue #187
  and task t4 of this plan (`lobes/roles.py` `ROLE_RESPONSIBILITIES`). Any
  consumer routing images/video or code-authoring work to `model=worker`
  needs to stop doing so independent of the id swap.
- **No dual-served-name decision has been made yet** for this swap
  (playbook §2 option 1). If chosen, update this note before the flip lands
  to say so — it changes every instruction above.
- **Re-run the audit close to the actual flip date.** This snapshot is dated
  2026-08-20; GitHub code search indexing lag and any new consumer added to
  the mesh between now and the live boot (t8) are not covered by these
  results.

## How to verify (once the flip has actually happened — not yet)

```bash
# expect the NEW id, 200:
curl -s http://<thor-gateway>:8000/v1/models | grep -o '"id":"[^"]*"'

# expect 200 (role alias still resolves):
curl -s -o /dev/null -w '%{http_code}\n' \
  http://<thor-gateway>:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"worker","messages":[{"role":"user","content":"hi"}],"max_tokens":1}'

# on a proxying box (Spark/Orin), expect the mirror to have landed:
lobes capabilities | grep -A3 worker
# ready=true / proxied=true / model=nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4
```

Expect: `/v1/models` on Thor lists the Lightning id (not the 3.6 id);
`worker` returns 200; a proxying box's `lobes capabilities` shows
`ready=true` only after its own `.env` mirror lands.

## Appendix: audit queries and raw results

Run 2026-08-20, via `gh api search/code`. GitHub code search is not
exhaustive (indexing lag, private repos, forks) — treat a zero-hit result as
"nothing found today," not as a permanent guarantee.

### Query 1 — org-wide, bare model family string

```bash
gh api search/code -X GET -f q='"Qwen3.6-35B-A3B" org:agentculture' \
  --jq '.items[] | .repository.full_name + ": " + .path'
```

Total hits: **178** (search API caps enumerable items; page 1 sample below).
By repo (page-1 distinct repos observed): `agentculture/lobes-cli`,
`agentculture/colleague`, `agentculture/reachy-mini-cli`,
`agentculture/embodiment`. No hits observed in `eidetic-cli`, `steward`, or
`guildmaster` in this or any other query (see queries 4–6).

### Query 2 — org-wide, exact old served id

```bash
gh api search/code -X GET -f q='"unsloth/Qwen3.6-35B-A3B-NVFP4" org:agentculture' \
  --jq '.items[] | .repository.full_name + ": " + .path'
```

Total hits: **152**. Same repo set as query 1
(`lobes-cli`, `colleague`, `reachy-mini-cli`, `embodiment`). Notable
`colleague` paths: `colleague/oilcheck/tool_calling.py` (benchmark-comment
table only), `colleague/oilcheck/three_tier.py` (discovers the model id at
runtime — see analysis above), `docs/experiments/2026-08-08-prove-self-learning-387-arms/*`,
`docs/experiments/2026-08-05-experiment-b-worker-promotion.md`,
`docs/features/three-tier.md`, `docs/live-testing.md` — all descriptive/
experimental-record prose, no hardcoded live config default found.
`reachy-mini-cli` path: `docs/specs/2026-08-01-embodiment-layer.md` —
prose only; `REACHY_EMBODY_WORKER_MODEL`'s actual default is the role name
`worker`, not this raw id (verified by reading
`docs/operating-reachy.md`'s reference row, not by this search).
`embodiment` path: `README.md` — a closed experiment report
(`worker-toolloop.md`, 2026-08-04), historical.

### Query 3 — org-wide, WORKER_SERVED_NAME env var name

```bash
gh api search/code -X GET -f q='"WORKER_SERVED_NAME" org:agentculture' \
  --jq '.items[] | .repository.full_name + ": " + .path'
```

Total hits: **14**, **all 14 in `agentculture/lobes-cli`** — `CHANGELOG.md`,
`lobes/gateway/_config.py`, `lobes/gateway/server.py`,
`lobes/templates/fleet/env.example`, `lobes/templates/fleet/docker-compose.yml`,
`tests/test_gateway_config_wiring.py`, `tests/test_worker_compose.py`,
`tests/test_gateway_proxy.py`, `tests/test_shape_contract_matrix.py`,
`tests/goldens/template-defaults.env`,
`tests/goldens/shapes/thor-worker__thor.env`,
`tests/goldens/shapes/thor-worker__spark.env`,
`.devague/deliveries/every-lobe-in-the-mesh-can-see-the-spark-reaches-t.json`,
`.eidetic/memory/lobes__public.jsonl`. **No external repo declares its own
`WORKER_SERVED_NAME`** — the env var itself is entirely internal to
lobes-cli's own templates/gateway. This is a clean result: no external `.env`
surface needs a `WORKER_SERVED_NAME` mirror by this name; the mirror
instruction above is about the peer's own `WORKER_SERVED_NAME` in **its own**
lobes-cli deployment `.env` (Spark, Orin), not a value any other repo's
source code declares.

### Query 4 — eidetic-cli, scoped

```bash
gh api search/code -X GET -f q='"Qwen3.6-35B-A3B" repo:agentculture/eidetic-cli' \
  --jq '.items[] | .path'
```

Total hits: **0**. Clean — no reference anywhere in `eidetic-cli`.

### Query 5 — steward, scoped

```bash
gh api search/code -X GET -f q='"Qwen3.6-35B-A3B" repo:agentculture/steward' \
  --jq '.items[] | .path'
```

Total hits: **0**. Clean.

### Query 6 — guildmaster, scoped

```bash
gh api search/code -X GET -f q='"Qwen3.6-35B-A3B" repo:agentculture/guildmaster' \
  --jq '.items[] | .path'
```

Total hits: **0**. Clean.

### Query 7 — this repo (lobes-cli), in-tree grep

```bash
grep -rn "Qwen3.6-35B-A3B" --include="*" . | grep -v "\.git/"
```

Result: dozens of hits (catalog, roles, gateway config, templates, tests,
goldens, docs, `.devague/`, `.eidetic/memory/`) — enumerated in full under
"In-repo (this checkout)" above. All owned by tasks t3/t9/t15 of this plan,
not this note.

```bash
grep -rln "WORKER_SERVED_NAME\|WORKER_PEER_ORIGIN" --include="*" . | grep -v "\.git/"
```

Result: `CLAUDE.md`, `CHANGELOG.md`, `lobes/gateway/_config.py`,
`lobes/gateway/server.py`, `lobes/templates/fleet/env.example`,
`lobes/templates/fleet/docker-compose.yml`, and the gateway/proxy/shape test
files + goldens listed above, plus `.devague/` and `.eidetic/memory/`
records. `WORKER_PEER_ORIGIN` itself is a knob this repo *defines and reads*
(`lobes/gateway/_config.py:194`, `_PEER_ORIGIN_ENV["worker"]`); it is not
tied to any specific served id and needs no change for this swap — only the
served-name value it sits beside on a proxying peer's `.env` does (see
"Peer `.env` mirrors" above).

## See also

- `docs/model-switch-playbook.md` §2 — the general "know who hardcodes the
  served id" playbook this note follows.
- `docs/qwen38-rollout-notes.md` — the precedent this note is modeled on,
  and the source of the 2026-08-05 peer-mirror lesson cited above.
- `docs/qwen3.6-35b-a3b-nvfp4.md` — the per-model doc for the outgoing
  checkpoint (stays in-tree as history/candidate story, task t3/t15).
- `docs/plans/2026-08-20-nemotron-lightning-worker.md` — this plan; see t3
  (catalog demotion), t9 (shape/env/compose commit), t15 (docs follow-up)
  for the in-repo surfaces this note deliberately does not edit.
- issue #187 — the worker responsibility-split rationale (text-only,
  non-coding).
