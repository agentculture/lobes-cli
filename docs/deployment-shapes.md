# Deployment shapes — which lobes a box hosts

A **deployment shape** answers a question the #108/#110 machine profile
deliberately does not: not "how is each role *tuned* on this card?" but
"which of the seven Colleague roles does *this box* host **at all**?" A shape
is composed as pure data **over** the machine profile at render time — the
two axes are orthogonal: **shape × card**. This document is the deep
reference; `lobes explain shapes` is the brief in-CLI version.

## What a deployment shape is

The fleet exposes seven first-class Colleague roles (issue #81): `cortex`,
`senses`, `muse`, `embedder`, `reranker`, `stt`, `tts` — `muse` being the
opt-in-hosted seventh (below). A shape declares the subset a
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

Four families exist:

- **machine-as-brain** (the default) — one box hosts every role its card can
  serve. This is today's behaviour, made explicit as data: a bare `lobes
  init` (no `--shape` at all) resolves this shape, and because it carries
  **zero overrides**, composing it changes nothing — the rendering is
  byte-identical to the pre-shape behaviour, muse or no muse: a non-hosted
  opt-in core role renders nothing at all (see "Opt-in core roles" below).
  A single-box operator makes no new decisions.
- **mesh-brain shapes** (`spark-lobe`, `thor-lobe`) — a box drops one heavy
  generate lobe to a peer box in the mesh and reclaims that lobe's freed
  GPU-memory budget for the lobe(s) it keeps. Opt-in, per box, via `lobes
  init --shape <name>`.
- **small-model reference shape** (`orin-small`) — a box with NEITHER heavy
  generate lobe at all, hosting the opt-in `minor` gear (`vllm-minor`)
  instead, plus the two pooling gears and the audio overlay. Declared as
  data only — **not validated on a physical Jetson AGX Orin** (the #108
  rule; see the support table below).
- **opt-in-core-role shape** (`thor-muse`) — a box that drops BOTH heavy
  default lobes and instead hosts `muse`, the opt-in creative/ideation lobe
  (Gemma 4 31B NVFP4), plus the two pooling gears and the audio overlay.
  Declared as data only — **no physical box has booted it** (the #108 rule;
  see the support table and "Opt-in core roles" below).

## The support table

| shape | hosts | status | validation |
|---|---|---|---|
| **machine-as-brain** (default) | `cortex`, `senses`, `embedder`, `reranker`, `stt`, `tts` — every role the card can serve | goldens | Zero overrides; composing it onto any card profile is a byte-identical no-op (pinned by `tests/goldens/shapes/` and `tests/test_shape_goldens.py`). This is the shape a bare `lobes init` has always rendered. |
| **spark-lobe** | `cortex`, `embedder`, `reranker`, `stt`, `tts` — drops `senses` | validated live | 2026-07-14 on the DGX Spark GB10 (`spark-f8a9`) — full acceptance run PASS: dropped-lobe honesty (4 phases), correctness probes (cortex known-answer, embedder, reranker), the advertised-implies-reachable gate (5/5), and the measured reclaimed budget. Transcript: `docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt`. |
| **thor-lobe** | `senses`, `embedder`, `reranker`, `stt`, `tts` — drops `cortex` | validated live | 2026-07-14 on the Jetson AGX Thor (`thor`) — full acceptance run PASS: dropped-lobe honesty, correctness probes (embedder, reranker, senses text known-answer), the advertised-implies-reachable gate (5/5), and the measured reclaimed budget. Transcript: `docs/evidence/2026-07-14-accept-thor-lobe-thor.txt`. |
| **orin-small** | `minor`, `embedder`, `reranker`, `stt`, `tts` — drops BOTH `cortex` and `senses` | **declared, UNVALIDATED** | Pure data, goldens-only (`tests/goldens/shapes/orin-small__{base,spark,thor}.env`, `tests/test_shape_goldens.py`). Ships for the Jetson AGX Orin 64GB reference target (mesh-brain end-state, issue #112, t2) mirroring `lobes/profiles/builtin/base.toml`'s own "conservative fallback for an unrecognised card" discipline exactly — **no physical Orin has booted this shape**, so it carries no live-validation row and no measured budget. Do not read this row as an "Orin is supported" claim; physical validation is its own follow-up. |
| **thor-muse** | `muse`, `embedder`, `reranker`, `stt`, `tts` — drops BOTH `cortex` and `senses`, hosts the opt-in `muse` lobe instead | **declared, UNVALIDATED** | Pure data (`lobes/profiles/builtin_shapes/thor-muse.toml`). Hosts the seventh Colleague role — `nvidia/Gemma-4-31B-IT-NVFP4`, the creative/ideation lobe — with the FULL muse declaration in its `[overrides.muse]` (see "Opt-in core roles" below). Its budget values (`gpu_mem_util=0.40`, `max_model_len=262144` — the full 256K native window) are **hypotheses**, not measurements — **no physical Thor has booted this shape**, so it carries no live-validation row and no measured budget; an acceptance run (`scripts/accept-shape.sh`) on a physical Thor is what would validate it. Do not read this row as a "muse is served" claim. See [`docs/gemma-4-31b-nvfp4.md`](gemma-4-31b-nvfp4.md). |

All shipped shapes are pure data over the `#108` `Profile` schema
— no per-shape Python branch exists anywhere in `lobes/profiles/shapes.py` or
`shape_render.py`; the five built-in TOML files
(`lobes/profiles/builtin_shapes/{machine-as-brain,spark-lobe,thor-lobe,orin-small,thor-muse}.toml`)
differ from each other only in their `hosts` role subset and their
`overrides` budget re-derivation. `orin-small` adds one
new hostable role beyond the seven first-class Colleague roles: the opt-in
`minor` gear (`lobes/profiles/shapes.py`'s `OPT_IN_ROLES`), which carries no
Profile knobs of its own — re-using the `cortex` role slot for a 4B model
instead would mean the box advertises the 27B Colleague role while actually
serving something else, which is exactly the half-honest posture #92
forbids. Hosting an opt-in gear also renders its **activation env**
(`shape_render.OPT_IN_ACTIVATION_ENV`): `COMPOSE_PROFILES=minor` un-gates the
profile-gated `vllm-minor` service and `MINOR_BASE_URL` /
`MINOR_SERVED_NAME` wire the gateway backend — without these three keys a
scaffolded `orin-small` would start no generate lane at all (found by review
on PR #121; pinned by `tests/test_shape_goldens.py` and the orin-small
goldens).

## Opt-in core roles: how `muse` is hosted

`muse` — the seventh Colleague role, the creative/ideation lobe
(`nvidia/Gemma-4-31B-IT-NVFP4`) — introduced a new shape concept
(`lobes/profiles/shapes.py`'s `OPT_IN_CORE_ROLES`): a role that carries the
**full per-machine Profile knob set** (the profile schema's core roles are now
`cortex`/`senses`/`muse`/`embedder`/`reranker`) yet is **never hosted by
machine-as-brain** — a 31B cannot co-reside with the default `cortex`+`senses`
duo on a 128 GB box. Concretely:

- **The machine-as-brain identity set is `DEFAULT_HOSTED_ROLES`** (the six:
  `cortex`/`senses`/`embedder`/`reranker`/`stt`/`tts`); the Colleague
  *contract* set capabilities reports (`COLLEAGUE_ROLES`) is seven. On every
  non-hosting shape — machine-as-brain included — muse renders *nothing*: the
  card's own muse declaration passes through verbatim, which is exactly what
  keeps machine-as-brain a byte-identical no-op over the bare card profile.
  A marker would be redundant anyway, since an unwired muse is already
  infeasible by default at the gateway (`OPT_IN_BACKENDS`). The one card that
  does emit `MUSE_FEASIBLE=false` is `base.toml`, through its own conservative
  veto — a card-level fact, not something the shape layer adds.
- **Hostable only by an explicit shape; the full declaration lives in that
  shape's own overrides**, not in a card profile: `thor-muse`'s
  `[overrides.muse]` carries the model, budget, quantization, and
  attention-backend knobs. The card profiles stay silent on muse, except
  `base.toml`, which vetoes it (`[roles.muse] feasible=false` — the
  conservative unknown-card rule).
- **Hosting muse renders its activation env**
  (`shape_render.OPT_IN_CORE_ACTIVATION_ENV` /
  `OPT_IN_CORE_COMPOSE_PROFILE`): `COMPOSE_PROFILES=muse` un-gates the
  profile-gated `vllm-muse` service in the base fleet template (parked like
  `vllm-minor`; same custom image as `vllm-multimodal`,
  `Dockerfile.vllm-gemma4`) and `MUSE_BASE_URL=http://vllm-muse:8000` wires
  the gateway backend — plus the `MUSE_*` knobs rendered from the shape's
  overrides via the ordinary profile-env path.
- **`MUSE_PEER_ORIGIN` / `MUSE_PEER_PROXY` / `MUSE_PEER_API_KEY` exist** just
  like every core role's referral/proxy channels, so a box that doesn't host
  muse can honestly refer (or transparently proxy) callers to the box that
  does.

## Selecting a shape

```bash
lobes init --shape <machine-as-brain|spark-lobe|thor-lobe|orin-small|thor-muse> [TARGET]
```

`orin-small` and `thor-muse` resolve and render exactly like the other three
(they are pure data, proven by the same goldens/tests) — but as of this
writing both are **declared, not validated**: nothing here or in `lobes
capabilities` claims a physical Orin has run `orin-small`, or that any
physical box has booted `thor-muse`.

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
  and `tts` identically (see the `hosts` list in every built-in TOML above), so
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
# (EMBED_PEER_ORIGIN / RERANK_PEER_ORIGIN — and MUSE_PEER_ORIGIN, for the
# opt-in muse lobe — exist too; stt/tts are outside the channel, exactly as
# they are outside *_FEASIBLE.)
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

**The default boundary: no data-plane proxying.** Declaring `*_PEER_ORIGIN`
alone is an annotation for the *caller* to act on — the gateway never forwards
a generate/embed/rerank/audio request to a peer on the strength of the origin
alone, never probes the declared origin on the request hot path, and a
request for an unhosted role terminates locally at the 404 with zero outbound
connections (test-enforced). A box that *follows* its own referral on the
caller's behalf — a proxy-lobe, advertised as proxied — is an explicit,
separate opt-in on top of the origin declaration: see
[Following the referral: proxy-lobes](#following-the-referral-proxy-lobes-opt-in)
below. With no `*_PEER_PROXY` armed anywhere (every deployment that predates
that feature, and every referral-only deployment today) this boundary holds
exactly as described here.

## Following the referral: proxy-lobes (opt-in)

Referral answers "who hosts this?"; proxy-lobes (issues #115/#127, phase 1)
answers the next question — "will you get it for me?" — with a third lobe
state on top of the two above:

| State | This box... | A request for the role gets |
|---|---|---|
| **awake** | hosts the role | served locally |
| **asleep** (referral-only) | dropped the role, named its peer | `404 role_infeasible` + `hosted_by: <peer origin>` — the caller must dial the peer itself |
| **proxy** | dropped the role, named its peer, *and* opted in to following the referral | forwarded to the peer; the caller never has to know it moved |

**The opt-in is a second, deliberate step — q1 from the #115/#127 design
work.** Declaring `<PREFIX>_PEER_ORIGIN` alone (above) stays **referral-only**
— origin without the proxy knob never gets dialed, preserving the issue #112
contract byte-for-byte. Setting the matching `<PREFIX>_PEER_PROXY=true` is
what additionally arms the gateway to **follow its own referral** on the
caller's behalf — and only for a name that is *also* infeasible on this box
*and* has that declared origin; the knob is inert on its own.

**The pairwise key contract.** Proxying introduces credentials in both
directions, and they are deliberately asymmetric:

- **One *inbound* key per box** — `GATEWAY_API_KEY` (fallback
  `CULTURE_VLLM_API_KEY`), the same knob [gateway auth](gateway-fleet.md#auth-opt-in-bearer-gate)
  uses to gate this box's own data plane.
- **One *outbound* key per dropped-role peer** — `<PREFIX>_PEER_API_KEY`. Its
  value is **the peer's own inbound key** (the credential *that box* requires
  from callers), never a value this box invents, and never this box's own
  `GATEWAY_API_KEY`.

Because the outbound credential is always a *copy* of the peer's existing
inbound key — never a secret freshly minted per relationship — key material
scales **O(machines)**: an N-box mesh needs N inbound keys total (one per
box), and a box proxying to M peers holds at most M copies of those peers'
own keys. **Keys never propagate through**: the caller's own `Authorization`
authenticated it to *this* box and is stripped before every forward; only the
declared pairwise key (or nothing, if none is declared) travels onward. This
is also why the referral/proxy origins assume a **tailnet-class transport**
(Tailscale, a VPN, or an otherwise-private/trusted network between boxes) —
the forward is plain HTTP with no TLS termination of its own, so
confidentiality in transit is the tailnet's job, exactly as it already is for
`CULTURE_VLLM_API_KEY` over `lobes tunnel`/cloudflared. Never point a
`*_PEER_ORIGIN` at a box reachable only over the public internet.

**Worked example — spark-lobe dropped `senses`; thor-lobe hosts it.** (Both
hostnames below are placeholders — substitute your own tailnet/VPN names,
never a real hostname or key in a committed file.)

```bash
# On the Spark box (spark-lobe: hosts cortex, dropped senses):
GATEWAY_API_KEY=<spark's own inbound key>
MULTIMODAL_PEER_ORIGIN=http://thor.example.ts.net:8000
MULTIMODAL_PEER_PROXY=true
MULTIMODAL_PEER_API_KEY=<thor's inbound GATEWAY_API_KEY — a copy, not a new secret>

# On the Thor box (thor-lobe: hosts senses), correspondingly:
GATEWAY_API_KEY=<thor's own inbound key — the SAME value Spark put in MULTIMODAL_PEER_API_KEY>
```

A `senses` chat request against Spark's gateway now: strips the caller's own
`Authorization`, attaches `Bearer <thor's key>`, forwards to
`http://thor.example.ts.net:8000`, and relays Thor's Gemma answer back with
`X-Lobes-Proxied-By: http://thor.example.ts.net:8000` — the caller never has
to know Thor exists. If Spark instead only set `MULTIMODAL_PEER_ORIGIN`
(no `_PEER_PROXY`, no key), the exact same request would still 404
`role_infeasible` naming Thor, as it did before proxy-lobes existed.

**Referral-only deployments are byte-identical to before.** Every honesty
surface, failure mode, and header this section describes is gated on
`<PREFIX>_PEER_PROXY` being armed for that specific role. A deployment that
never sets it — including both live `spark-lobe`/`thor-lobe` boxes as of this
writing (referral-only) — sees no behavior change at all: same 404 body, same
`hosted_by` annotation, same zero outbound connections.

See [`docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in`](gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in)
for the data-plane mechanics (marker headers, the single-hop loop guard, peer
failure modes, pressure semantics) and
[`docs/colleague-stack.md`](colleague-stack.md#a-third-role-state-proxied) for
how a proxied role shows up in the role contract.

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
scripts/accept-shape.sh <machine-as-brain|spark-lobe|thor-lobe|thor-muse|orin-small> [--audio] \
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
   By default, no data-plane proxying: a consumer dials each box directly,
   exactly as the Culture mesh does today; with opt-in peer config, a box's
   `capabilities` and its `role_infeasible` 404s name the peer that hosts an
   absent role (`hosted_by`, above). The #92 invariant holds throughout — a
   box never serves what it does not host. *Following* a referral on the
   caller's behalf (a proxy-lobe) was a deliberate non-goal **at the time
   this decision was recorded** — it has since landed as its own opt-in
   extension (issues #115/#127, phase 1): see
   [Following the referral: proxy-lobes](#following-the-referral-proxy-lobes-opt-in)
   above. Direct addressing remains the default; proxying is additive, never
   a replacement for it.
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

**What's still open:** proxy-lobes' phase-1 substrate (config channels,
inbound auth, the data-plane forward, the PROXIED capabilities state — see
[Following the referral: proxy-lobes](#following-the-referral-proxy-lobes-opt-in)
above) has landed, but its **live cross-box acceptance run** — the same kind
of physical-pair proof the referral evidence above already has — has not; no
doc here claims a proxied answer has been observed on physical hardware yet.
Issue #127's own later phases (fan-out execution, a request-tracing store,
latency-aware routing, policy plugins) remain explicitly out of scope for
this delivery. Physical Jetson AGX Orin 64GB validation (decision 3's
`orin-small` reference instance — declared, not yet booted) is also still
open; see the support table above and the scope boundary below.

## Scope boundary

The near-term work (issue #113) shipped shape selection plus `spark-lobe` and
`thor-lobe` (both validated live). The mesh-brain end-state (issue #112,
spec+plan in PR #116) has since landed on top of it: the `orin-small`
reference shape, the opt-in honest-referral surface, the per-(shape,
dropped-role) contract-test matrix, and the live cross-box referral evidence
above are all shipped. Two things remain explicitly out of scope, routed
elsewhere:

- **Proxy-lobes** (serving a "sleeping" lobe by following its own referral to
  whichever peer box hosts it, on the caller's behalf) — the phase-1
  substrate has shipped as an opt-in extension (issues #115/#127; see
  [Following the referral: proxy-lobes](#following-the-referral-proxy-lobes-opt-in)
  above). Direct addressing + referral (decision 1, above) remains the
  shipped **default**; a live cross-box acceptance run for the proxy path,
  and #127's later phases (fan-out execution, request tracing, latency-aware
  routing, policy plugins), are their own follow-ups, out of scope here.
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
- `docs/specs/2026-07-16-proxy-lobes-pairwise-auth.md` /
  `docs/plans/2026-07-16-proxy-lobes-pairwise-auth.md` — the proxy-lobes +
  pairwise-auth spec and plan (#115/#127 phase 1): the awake/asleep/proxy
  table, the pairwise credential model, the tailnet transport assumption, and
  the task-by-task delivery this document's [Following the referral](#following-the-referral-proxy-lobes-opt-in)
  section describes
- `docs/evidence/2026-07-14-accept-referral-thor.txt` — the live cross-box
  referral evidence (#112, t5); proxy-lobes' own live cross-box evidence is
  still open (see "What's still open" above)
- `docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in` — the
  data-plane mechanics: marker headers, the loop guard, peer failure modes
- `docs/machine-profiles.md` — the per-machine (card) tuning axis this
  composes with
- `docs/colleague-stack.md` — the seven-role Colleague contract, including the
  proxied role state
- `lobes/profiles/shapes.py` — the `Shape` schema + built-in loader
  (`COLLEAGUE_ROLES` / `DEFAULT_HOSTED_ROLES` / `OPT_IN_CORE_ROLES` /
  `OPT_IN_ROLES` / `SHAPE_ROLES`)
- `lobes/profiles/shape_render.py` — the pure `(shape, profile) → compose/.env`
  renderer
- `lobes/profiles/builtin_shapes/*.toml` — the five shipped shapes, with
  provenance comments on every override (for `orin-small`, on why its
  generate lane is `minor` rather than `cortex`; for `thor-muse`, on why the
  full `muse` declaration lives in the shape rather than a card profile)
- `lobes/cli/_commands/init.py` — the `--shape` flag + generated
  `docker-compose.shape.yml` override
- `scripts/accept-shape.sh` — the acceptance script
- `docs/evidence/2026-07-14-accept-spark-lobe-gb10.txt` /
  `2026-07-14-accept-thor-lobe-thor.txt` — the live validation transcripts
