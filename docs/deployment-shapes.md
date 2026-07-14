# Deployment shapes — which lobes a box hosts

A **deployment shape** answers a question the #108/#110 machine profile
deliberately does not: not "how is each role *tuned* on this card?" but
"which of the six Colleague roles does *this box* host **at all**?" A shape
is composed as pure data **over** the machine profile at render time — the
two axes are orthogonal: **shape × card**. This document is the deep
reference; `lobes explain shapes` is the brief in-CLI version.

## What a deployment shape is

The fleet exposes six first-class Colleague roles (issue #81): `cortex`,
`senses`, `embedder`, `reranker`, `stt`, `tts`. A shape declares the subset a
box **hosts** (`lobes/profiles/shapes.py`'s `Shape.hosts`) plus, optionally, a
per-role budget **override** that re-derives `gpu_mem_util` /
`max_model_len` for a role that no longer shares the box with a dropped one
(`Shape.overrides`, reusing the same `RoleProfile` knob vocabulary the card
profile itself uses — never a parallel, re-typed schema).

Rendering composes a `(shape, profile)` pair into the concrete `.env` +
compose file list (`lobes/profiles/shape_render.py::render_shape`) — a role
the shape hosts gets the card's tuning with the shape's override overlaid; a
role the shape drops renders the #110-conventional `<PREFIX>_FEASIBLE=false`
marker and nothing else. This is a **pure function of (shape, profile,
template)**: no GPU probe, no host read, no subprocess, so it runs
identically on a GPU-less CI runner.

Two families exist:

- **machine-as-brain** (the default) — one box hosts every role its card can
  serve. This is today's behaviour, made explicit as data: a bare `lobes
  init` (no `--shape` at all) resolves this shape, and because it carries
  **zero overrides**, composing it changes nothing — the rendering is
  byte-identical to the pre-shape behaviour. A single-box operator makes no
  new decisions.
- **mesh-brain shapes** (`spark-lobe`, `thor-lobe`) — a box drops one heavy
  generate lobe to a peer box in the mesh and reclaims that lobe's freed
  GPU-memory budget for the lobe(s) it keeps. Opt-in, per box, via `lobes
  init --shape <name>`.

## The support table

| shape | hosts | status | validation |
|---|---|---|---|
| **machine-as-brain** (default) | `cortex`, `senses`, `embedder`, `reranker`, `stt`, `tts` — every role the card can serve | goldens | Zero overrides; composing it onto any card profile is a byte-identical no-op (pinned by `tests/goldens/shapes/` and `tests/test_shape_goldens.py`). This is the shape a bare `lobes init` has always rendered. |
| **spark-lobe** | `cortex`, `embedder`, `reranker`, `stt`, `tts` — drops `senses` | validated live | 2026-07-14 on the DGX Spark GB10 (`spark-f8a9`) — full acceptance run PASS: dropped-lobe honesty (4 phases), correctness probes (cortex known-answer, embedder, reranker), the advertised-implies-reachable gate (5/5), and the measured reclaimed budget. Transcript: `docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt`. |
| **thor-lobe** | `senses`, `embedder`, `reranker`, `stt`, `tts` — drops `cortex` | validated live | 2026-07-14 on the Jetson AGX Thor (`thor`) — full acceptance run PASS: dropped-lobe honesty, correctness probes (embedder, reranker, senses text known-answer), the advertised-implies-reachable gate (5/5), and the measured reclaimed budget. Transcript: `docs/evidence/2026-07-14-accept-thor-lobe-thor.txt`. |

Both mesh-lobe shapes are pure data over the shipped `#108` `Profile` schema
— no per-shape Python branch exists anywhere in `lobes/profiles/shapes.py` or
`shape_render.py`; the three built-in TOML files
(`lobes/profiles/builtin_shapes/{machine-as-brain,spark-lobe,thor-lobe}.toml`)
differ from each other only in their `hosts` role subset and (for the two
mesh shapes) their `overrides` budget re-derivation.

## Selecting a shape

```bash
lobes init --shape <machine-as-brain|spark-lobe|thor-lobe> [TARGET]
```

- **Dry-run by default** — prints the resolved profile, the shape and its
  `hosts` list, how many env vars would be set, and (for a mesh shape)
  whether `docker-compose.shape.yml` would be written or removed. Changes
  zero bytes.
- **`--apply`** commits: renders the `(shape, profile)` pair's env into
  `.env`, persists `LOBES_PROFILE` / `MODEL_GEAR_VERSION`, and writes (or
  scrubs) the generated shape override.
- **Byte-for-byte restore** — re-running `lobes init --shape <previous>
  --apply` restores the previous rendering exactly:
  `tests/test_init_shape.py::test_reapplying_shape_is_idempotent_including_override`
  and its sibling for a bare re-init both pin this. Re-initialising to
  `machine-as-brain` over a mesh-shape scaffold also **scrubs** the stale
  `docker-compose.shape.yml` — otherwise `docker compose up` would keep
  skipping a lobe the new shape re-hosts.
- **`--single` conflict** — `--shape` is a fleet-scaffold axis (the legacy
  single-model topology never resolves a profile or a shape at all), so
  passing `--shape` together with `--single` is a hard user error, even when
  the shape named is the default `machine-as-brain`.
- **`--shape` × `--audio`** — independent. Every built-in shape hosts `stt`
  and `tts` identically (see the `hosts` list in all three TOMLs above), so
  `--audio` is the sole switch that scaffolds the realtime audio overlay
  (`docker-compose.audio.yml`); passing both flags together is harmless and
  idempotent. A dropped core lobe never affects whether audio is scaffolded,
  and vice versa.
- **Unknown `--shape` value** — a user error naming the valid (sorted)
  shapes; resolved *before* anything is written, in both dry-run and
  `--apply`.

## What a mesh shape does end-to-end

Dropping a lobe is honest at every layer a caller can observe — never
half-served, never silently rerouted:

1. **The service does not run.** `lobes init --shape <mesh-shape> --apply`
   generates `docker-compose.shape.yml`, a compose *override* layered last
   (`-f docker-compose.yml [-f docker-compose.audio.yml] -f
   docker-compose.shape.yml`, which `lobes fleet up --apply` auto-includes
   when present). It parks each dropped core service in the inert
   `shape-dropped` compose profile — a profile **nothing** activates, so
   `docker compose up` skips it — and clears the gateway's `depends_on` with
   the compose `!reset` merge tag (list *replacement* is `!override`; `!reset`
   is what removes the now-dangling edge to a profile-disabled service).
   **Requires Docker Compose v2.24+** (the `!reset` tag).
2. **Capabilities flag it, on both surfaces.** The dropped role reports
   `feasible: false` in `lobes capabilities` **and** the gateway's `GET
   /capabilities` — verified to agree live in the acceptance runs.
3. **`/v1/models` omits it entirely.** No alias for the dropped role appears
   in the model list.
4. **Every alias 404s `role_infeasible`.** A request naming the dropped role
   by any of its aliases — the role name itself, the capability-tier alias,
   and the back-compat alias (e.g. `senses` / `multimodal` / `normal` on
   `spark-lobe`; `cortex` / `main` / `hard` on `thor-lobe`) — gets `404
   role_infeasible`, never a silent reroute to a different model.
5. **`lobes up <dropped-role>` is a user error naming the shape.** Rather
   than letting `docker compose` fail with an opaque "no such service",
   `lobes up` reads the shape override, detects the target needs a dropped
   service, and errors with the override file and a remediation pointing at
   re-scaffolding with a shape that hosts it.

## The co-residency tax and its measured repayment

**Before** (machine-as-brain, the default, on the GB10): every role
co-resides, so each one is trimmed to fit alongside the others —
`senses` 32768 @ util 0.14, `cortex` 131072 @ util 0.30, embedder/reranker
at 0.06 each — total budget `0.30 + 0.14 + 0.06 + 0.06 = 0.56`.

**spark-lobe** (measured live, 2026-07-14): dropping `senses` lets `cortex`
reclaim budget. `PRIMARY_GPU_MEM_UTIL=0.44` / `PRIMARY_MAX_MODEL_LEN=262144`
— its full native 256K context. Measured KV pool: **888,946 tokens**,
**3.39×** concurrency at the full 256K request length. The historical 0.60
solo value was **tried first and refused by vLLM on the live box**: the
GB10's 121.7 GiB is *unified* memory shared with the host OS and other
services, and the 2026-07-14 boot measured only **59.35 GiB free** at
primary startup against the **73.01 GiB** that util 0.60 demands. 0.44
(53.5 GiB) fits the measured reality with margin.

**thor-lobe** (measured live, 2026-07-14): dropping `cortex` lets `senses`
reclaim budget. `MULTIMODAL_GPU_MEM_UTIL=0.30` /
`MULTIMODAL_MAX_MODEL_LEN=131072` — its full native 128K context. Measured KV
pool: **1,418,554 tokens**, **10.82×** concurrency at 131072. The
reclaim-**sum** 0.14 (senses' own co-resident share) + 0.30 (dropped
cortex's share) = 0.44 was likewise tried first and **refused**: the live
Thor measured only **38.44 GiB free** at senses startup against the
**54.04 GiB** that util 0.44 demands (Thor's unified memory carries heavier
host workloads than the Spark's). 0.30 (36.85 GiB) is exactly the dropped
cortex's own freed share, and it fits.

**The lesson, stated plainly:** on unified-memory boxes the reclaim values
are **measured truths, not arithmetic** — a naive "sum the freed shares" or
"promote to the model's own solo default" both looked reasonable on paper
and both were refused by vLLM on the physical box. The acceptance run
(`scripts/accept-shape.sh`, below) is what validates a shape's budget on a
given box; the shipped TOML overrides carry the measured value plus its
provenance comment, not an estimate.

## Before-state: the four core roles were unconditional

The claim this feature starts from is checkable in the shipped tree: in
`lobes/templates/fleet/docker-compose.yml`, the four core-role services
(`vllm-primary`, `vllm-embed`, `vllm-rerank`, `vllm-multimodal`) carry **no**
`profiles:` stanza — only the opt-in gears (`vllm-minor`, `vllm-middle`,
`vllm-multimodal-coder`) do. Before this feature, "Spark drops Gemma" or
"Thor drops Qwen" meant hand-editing `docker-compose.yml`, exactly the drift
the #108 goldens exist to prevent. Shapes gate a dropped service via the
**generated** `docker-compose.shape.yml` override layered on top — the base
template itself stays unconditional, so `machine-as-brain` (the default)
needs no override at all. Issue #109 (the GB10 verification this work
depended on) is closed, with the GB10 trait findings recorded on the issue.

## The dev lane

`GATEWAY_PIP_EXTRA_INDEX_URL` in `.env` (default unset — a no-op on release
builds) paired with a TestPyPI `.devN` `MODEL_GEAR_VERSION` lets a
from-source box validate an unreleased branch end-to-end, with zero hand
edits to the Dockerfiles. The gateway and realtime images install in **two
steps**: first `pip install --no-cache-dir --no-deps --index-url
"${LOBES_DEV_INDEX_URL}" "lobes-cli==${MODEL_GEAR_VERSION}"`, then the normal
`pip install "lobes-cli==${MODEL_GEAR_VERSION}"` against real PyPI (which
sees the pin already satisfied). The two-step, `--no-deps`-first shape exists
because a plain extra-index install lets a TestPyPI name-squat (e.g.
`FASTAPI-1.0`) outrank the real PyPI package of the same name — fetching the
`.devN` wheel `--no-deps` from the dev index alone, then resolving every
other dependency from real PyPI, avoids that entirely.

## The acceptance script

```bash
scripts/accept-shape.sh <machine-as-brain|spark-lobe|thor-lobe> [--audio] \
  [--deploy-dir DIR] [--port N] [--env KEY=VAL] [--dev-version V] \
  [--dev-index URL] [--timeout SECS]
scripts/accept-shape.sh --restore [--deploy-dir DIR]
```

One unattended, fail-not-skip command: back up the current deployment,
scaffold the requested shape into a clean dir (dry-run shown first, then
`--apply`), boot the fleet, prove dropped-lobe honesty and per-role
correctness **live**, run the advertised-implies-reachable gate
(`scripts/live-check.sh`), measure the reclaimed heavy-lobe budget, and leave
a full transcript at `~/lobes-accept-<shape>-<UTC-stamp>.log`. `--restore`
puts the previous deployment back. This is the exact command sequence behind
both validation transcripts in `docs/evidence/`:
`2026-07-14-accept-spark-lobe-gb10.txt` and
`2026-07-14-accept-thor-lobe-thor.txt`.

## Scope boundary

This feature ships shape selection plus the two near-term mesh-lobe shapes
only. Explicitly out of scope, routed elsewhere:

- **One-lobe-per-box end-state** (a Qwen-only Spark, a Gemma-only Thor, small
  models on an Orin) and the **cross-box referral** story (a box's gateway
  proxying a request to the peer that actually hosts the dropped role,
  rather than today's per-box-honesty-only contract) — tracked as issue
  #112, with its own spec + plan opened in PR #116.
- **Proxy-lobes** (serving a "sleeping" lobe via a followed referral to
  whichever peer box hosts it) — parked as its own follow-up, issue #115.
- **Jetson AGX Orin** — named in the mesh topology as a future target but
  unvalidated; no profile or shape ships for it here.

## See also

- `lobes explain shapes` — brief in-CLI reference
- `docs/machine-profiles.md` — the per-machine (card) tuning axis this
  composes with
- `docs/colleague-stack.md` — the six-role Colleague contract
- `lobes/profiles/shapes.py` — the `Shape` schema + built-in loader
- `lobes/profiles/shape_render.py` — the pure `(shape, profile) → compose/.env`
  renderer
- `lobes/profiles/builtin_shapes/*.toml` — the three shipped shapes, with
  provenance comments on every override
- `lobes/cli/_commands/init.py` — the `--shape` flag + generated
  `docker-compose.shape.yml` override
- `scripts/accept-shape.sh` — the acceptance script
- `docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt` /
  `2026-07-14-accept-thor-lobe-thor.txt` — the live validation transcripts
