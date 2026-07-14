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

Three families exist:

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
- **small-model reference shape** (`orin-small`) — a box with NEITHER heavy
  generate lobe at all, hosting the opt-in `minor` gear (`vllm-minor`)
  instead, plus the two pooling gears and the audio overlay. Declared as
  data only — **not validated on a physical Jetson AGX Orin** (the #108
  rule; see the support table below).

## The support table

| shape | hosts | status | validation |
|---|---|---|---|
| **machine-as-brain** (default) | `cortex`, `senses`, `embedder`, `reranker`, `stt`, `tts` — every role the card can serve | goldens | Zero overrides; composing it onto any card profile is a byte-identical no-op (pinned by `tests/goldens/shapes/` and `tests/test_shape_goldens.py`). This is the shape a bare `lobes init` has always rendered. |
| **spark-lobe** | `cortex`, `embedder`, `reranker`, `stt`, `tts` — drops `senses` | validated live | 2026-07-14 on the DGX Spark GB10 (`spark-f8a9`) — full acceptance run PASS: dropped-lobe honesty (4 phases), correctness probes (cortex known-answer, embedder, reranker), the advertised-implies-reachable gate (5/5), and the measured reclaimed budget. Transcript: `docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt`. |
| **thor-lobe** | `senses`, `embedder`, `reranker`, `stt`, `tts` — drops `cortex` | validated live | 2026-07-14 on the Jetson AGX Thor (`thor`) — full acceptance run PASS: dropped-lobe honesty, correctness probes (embedder, reranker, senses text known-answer), the advertised-implies-reachable gate (5/5), and the measured reclaimed budget. Transcript: `docs/evidence/2026-07-14-accept-thor-lobe-thor.txt`. |
| **orin-small** | `minor`, `embedder`, `reranker`, `stt`, `tts` — drops BOTH `cortex` and `senses` | **declared, UNVALIDATED** | Pure data, goldens-only (`tests/goldens/shapes/orin-small__{base,spark,thor}.env`, `tests/test_shape_goldens.py`). Ships for the Jetson AGX Orin 64GB reference target (mesh-brain end-state, issue #112, t2) mirroring `lobes/profiles/builtin/base.toml`'s own "conservative fallback for an unrecognised card" discipline exactly — **no physical Orin has booted this shape**, so it carries no live-validation row and no measured budget. Do not read this row as an "Orin is supported" claim; physical validation is its own follow-up. |

Both mesh-lobe shapes are pure data over the shipped `#108` `Profile` schema
— no per-shape Python branch exists anywhere in `lobes/profiles/shapes.py` or
`shape_render.py`; the four built-in TOML files
(`lobes/profiles/builtin_shapes/{machine-as-brain,spark-lobe,thor-lobe,orin-small}.toml`)
differ from each other only in their `hosts` role subset and (for the two
mesh shapes) their `overrides` budget re-derivation. `orin-small` adds one
new hostable role beyond the six first-class Colleague roles: the opt-in
`minor` gear (`lobes/profiles/shapes.py`'s `OPT_IN_ROLES`), which carries no
Profile knobs of its own — re-using the `cortex` role slot for a 4B model
instead would mean the box advertises the 27B Colleague role while actually
serving something else, which is exactly the half-honest posture #92
forbids.

## Selecting a shape

```bash
lobes init --shape <machine-as-brain|spark-lobe|thor-lobe|orin-small> [TARGET]
```

`orin-small` resolves and renders exactly like the other three (it is pure
data, proven by the same goldens/tests) — but as of this writing it is
**declared, not validated**: nothing here or in `lobes capabilities` claims a
physical Orin has run it.

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

## Honest referral to the peer that hosts a dropped role (opt-in)

A box that dropped a role can additionally *tell callers who does host it* —
the confirmed cross-box decision for the mesh-brain end-state (issue #112):
**direct + referral**. Consumers address boxes directly, exactly as the
Culture mesh does today; a box that doesn't host a role answers honestly with
who does.

**The peer-config surface** is one env var per core role's backend in the
deployment's `.env`, mirroring the `*_FEASIBLE` flags (`PEER_ORIGIN_ENV` in
`lobes/gateway/_config.py`):

```bash
# thor-lobe dropped cortex; the Spark hosts it:
PRIMARY_PEER_ORIGIN=http://spark.local:8001
# spark-lobe dropped senses; the Thor hosts it:
MULTIMODAL_PEER_ORIGIN=http://thor.local:8001
# (EMBED_PEER_ORIGIN / RERANK_PEER_ORIGIN exist too; stt/tts are outside the
# channel, exactly as they are outside *_FEASIBLE.)
```

The origin is a full, **operator-declared** URL. It is never derived from
hostnames, interfaces, or anything the box could guess — the #92 lesson:
never fabricate an absolute URL. Declaring an origin for a role the box
*does* host annotates nothing (a referral names who hosts what this box does
not).

**What it changes** — the two honesty surfaces only:

- `lobes capabilities` / `GET /capabilities`: the unhosted role's entry gains
  `"hosted_by": "<peer origin>"` next to its `feasible: false`.
- The `404 role_infeasible` body: the error object gains the same
  `"hosted_by"` key and the message names the peer origin.

`/v1/models` is untouched (it still simply omits the unhosted role), and with
**zero peer config — the default — every response is byte-identical to the
pre-referral contract** (regression-pinned in `tests/test_peer_referral.py`).

**The boundary: no data-plane proxying.** The referral is an annotation for
the *caller* to act on — the gateway never forwards a generate/embed/rerank/
audio request to a peer, never probes the declared origin on the request hot
path, and a request for an unhosted role terminates locally at the 404 with
zero outbound connections (test-enforced). A box that *follows* its own
referral on the caller's behalf — a proxy-lobe, advertised as proxied — is a
deliberate non-goal here, deferred to issue #115.

## The co-residency tax and its measured repayment

| box (mesh shape) | heavy lobe | co-resident context (machine-as-brain) | full-native context | repayment | measured `gpu_mem_util` | measured KV pool | measured concurrency |
|---|---|---|---|---|---|---|---|
| Spark GB10 (`spark-lobe`) | `cortex` | 131072 | 262144 | **2.0×** | 0.44 | **888,946 tokens** | **3.39×** at full 256K |
| Jetson AGX Thor (`thor-lobe`) | `senses` | 32768 | 131072 | **4.0×** | 0.30 | **1,418,554 tokens** | **10.82×** at 131072 |

All eight numbers above are read verbatim from the two `#113` acceptance
transcripts (`docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt` phase 7,
`docs/evidence/2026-07-14-accept-thor-lobe-thor.txt` phase 7) plus the
deployed `.env` overrides they measured — none are estimated.

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
scripts/accept-shape.sh <machine-as-brain|spark-lobe|thor-lobe|orin-small> [--audio] \
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
puts the previous deployment back. Phase 4b additionally checks the opt-in
honest-referral surface (mesh-brain t3, issue #112) whenever a
`<PREFIX>_PEER_ORIGIN` is declared on the box under test — skipped, not
failed, with zero peer config. This is the exact command sequence behind the
two `#113` shape-validation transcripts in `docs/evidence/`:
`2026-07-14-accept-spark-lobe-gb10.txt` and
`2026-07-14-accept-thor-lobe-thor.txt`. The cross-box referral proof (a
consumer actually reaching the peer that hosts a dropped role) needs two live
boxes at once and so was run as a bespoke variant rather than a single
`accept-shape.sh` invocation — see
`docs/evidence/2026-07-14-accept-referral-thor.txt` and "The mesh-brain
end-state" below.

## The mesh-brain end-state (issue #112)

The near-term spec that shipped shape selection plus `spark-lobe`/`thor-lobe`
(`docs/specs/2026-07-14-lobes-serves-the-brain-shape-you-choose-machine-as.md`)
deliberately left the cross-box question open. Its Decisions section says so
explicitly:

> Cross-box story is per-box honesty only for this change: each box
> advertises only the lobes it hosts and consumers address each box directly
> (how the Culture mesh already connects per machine); gateway proxying of
> absent roles and a brain-level capabilities view are deferred to the #112
> design work.

Issue #112's own exported spec
(`docs/specs/2026-07-14-lobes-serves-the-mesh-brain-end-state-one-lobe-per.md`)
is the answer to that deferral. Its framing, in one line: **one heavy lobe
per box, cheap gears co-reside, and the brain stays whole across the mesh via
direct addressing plus honest referral.** Four decisions came out of it, all
now shipped:

1. **Cross-box reachability = direct addressing + opt-in honest referral.**
   No data-plane proxying: a consumer dials each box directly, exactly as the
   Culture mesh does today; with opt-in peer config, a box's `capabilities`
   and its `role_infeasible` 404s name the peer that hosts an absent role
   (`hosted_by`, above). The #92 invariant holds throughout — a box never
   serves what it does not host. *Following* a referral on the caller's
   behalf (a proxy-lobe) is a deliberate non-goal here, deferred to issue
   #115.
2. **Cheap-gear placement = co-residence.** "One lobe per box" specializes
   the heavy *generate* lobes only — `embedder` / `reranker` / `stt` / `tts`
   may ride on every box that wants them (~0.06 util each, as today); no
   gear is forced to move for the end-state to hold, and consumers keep
   localhost embed/rerank/audio endpoints on every box.
3. **The fleet's reference shape assignment.** The Spark GB10 gives its
   whole machine to the Qwen `cortex` (shape `spark-lobe` — see the tax table
   above: it already reaches `cortex`'s full native 262144 context, not a
   partial reclaim); the Jetson AGX Thor 128GB gives its whole machine to the
   Gemma `senses` (shape `thor-lobe` — likewise reaches `senses`'s full
   native 131072); the Jetson AGX Orin 64GB hosts the small-model lobes
   (shape `orin-small` — the opt-in `minor` gear plus the pooling gears, no
   heavy 27B/12B at all). The two near-term shapes turned out to already *be*
   their one-lobe-per-box reference instances once their reclaim was
   measured; `orin-small` is the third reference instance, added by this
   work as declared-but-unvalidated data (support table above).
4. **The shape axis is mixable.** Backward compatible, just a design option:
   either many machines (some cloud, if you want) each specialized to one
   lobe, or some machines taking multiple roles, or any mix of the two —
   `machine-as-brain` stays the default and the one-lobe shapes are the far
   end of the same shape axis, not a mandate. A single-box operator running
   bare `lobes init` is completely unaffected.

**Live evidence for the cross-box surface.** Decisions 1–2 (referral +
co-residence) were validated live on the physical Jetson AGX Thor
(2026-07-14, `docs/evidence/2026-07-14-accept-referral-thor.txt`), dialing a
real declared peer (`PRIMARY_PEER_ORIGIN=http://spark.tail0be7e0.ts.net:8001`,
the box that hosts `cortex`) from a from-source gateway on the Thor with
`cortex` dropped. It proved, live: `capabilities` flags `cortex
feasible:false` and carries `hosted_by`, not hidden; every dropped-`cortex`
alias 404s `role_infeasible` with the same `hosted_by`; the hosted roles
(`senses`, `embedder`) answer through the same gateway; a consumer that
follows the referral and dials the peer directly reaches `cortex` there
(`'cortex-on-spark-alive'`); and a shape move is byte-for-byte restorable
(`machine-as-brain` → `thor-lobe` → `machine-as-brain`, tree hashes
`965708cc23da1ea5…` / `0c8e921fd3b27689…` / `965708cc23da1ea5…`). The
full-shape boots, per-role correctness probes, and measured reclaimed budgets
that decision 3 depends on were **not** re-run for this evidence — they reuse
the `#113` acceptance transcripts already cited in the tax table above (an
explicit operator decision recorded in the transcript itself, since spinning
up both physical boxes again would prove nothing new).

That referral run also surfaced one honest, unrelated finding: the Thor box's
*long-running* machine-as-brain deployment — the one left up to host the
`senses`/`embedder`/`reranker` lanes the referral test dialed through —
predates the `#110` per-machine-profile work, so its `.env` carries the
pre-`#110` reranker knobs and its rerank lane hangs. This is deployment
staleness on that one box, not a defect of the shapes or referral feature:
`docs/evidence/2026-07-14-accept-thor-lobe-thor.txt` (a freshly-scaffolded
`thor-lobe` deployment, `#113`) already shows the reranker probe passing
under the correct `RERANK_ENFORCE_EAGER`/`TRITON_ATTN` knobs. The fix, when
that box is next touched, is simply to re-scaffold with the current release.

**What's still open:** proxy-lobes (issue #115, decision 1's explicit
non-goal) and physical Jetson AGX Orin 64GB validation (decision 3's
`orin-small` reference instance — declared, not yet booted; see the support
table above and the scope boundary below).

## Scope boundary

The near-term work (issue #113) shipped shape selection plus `spark-lobe` and
`thor-lobe` (both validated live). The mesh-brain end-state (issue #112,
spec+plan in PR #116) has since landed on top of it: the `orin-small`
reference shape, the opt-in honest-referral surface, the per-(shape,
dropped-role) contract-test matrix, and the live cross-box referral evidence
above are all shipped. Two things remain explicitly out of scope, routed
elsewhere:

- **Proxy-lobes** (serving a "sleeping" lobe by following its own referral to
  whichever peer box hosts it, on the caller's behalf) — parked as its own
  follow-up, issue #115. Direct addressing + referral (decision 1, above) is
  the shipped default; proxying is a deliberate non-goal here.
- **Jetson AGX Orin physical validation** — the `orin-small` shape (above)
  ships as **declared, unvalidated data** (issue #112, mesh-brain end-state
  t2), exactly like `lobes/profiles/builtin/base.toml`'s existing
  conservative fallback for an unrecognised card: pure TOML + goldens, no
  live boot. Physical Jetson AGX Orin 64GB validation is its own follow-up
  with its own evidence — until it lands, no doc, support table, or `lobes
  capabilities` output may claim Orin is validated.

## See also

- `lobes explain shapes` — brief in-CLI reference
- `docs/specs/2026-07-14-lobes-serves-the-brain-shape-you-choose-machine-as.md`
  — the near-term (#113) spec: machine-as-brain default, `spark-lobe` /
  `thor-lobe`, per-box-honesty-only cross-box decision deferred to #112
- `docs/specs/2026-07-14-lobes-serves-the-mesh-brain-end-state-one-lobe-per.md`
  — the mesh-brain end-state (#112) spec: the four decisions above
- `docs/evidence/2026-07-14-accept-referral-thor.txt` — the live cross-box
  referral evidence (#112, t5)
- `docs/machine-profiles.md` — the per-machine (card) tuning axis this
  composes with
- `docs/colleague-stack.md` — the six-role Colleague contract
- `lobes/profiles/shapes.py` — the `Shape` schema + built-in loader
  (`COLLEAGUE_ROLES` / `OPT_IN_ROLES` / `SHAPE_ROLES`)
- `lobes/profiles/shape_render.py` — the pure `(shape, profile) → compose/.env`
  renderer
- `lobes/profiles/builtin_shapes/*.toml` — the four shipped shapes, with
  provenance comments on every override (and, for `orin-small`, on why its
  generate lane is `minor` rather than `cortex`)
- `lobes/cli/_commands/init.py` — the `--shape` flag + generated
  `docker-compose.shape.yml` override
- `scripts/accept-shape.sh` — the acceptance script
- `docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt` /
  `2026-07-14-accept-thor-lobe-thor.txt` — the live validation transcripts
