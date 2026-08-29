# The deployment lock and the variation catalog

A box's deployment is the compose files, overrides and Dockerfiles it
*actually runs* — not the packaged templates it was scaffolded from months
ago. Nothing tracked the divergence between the two until this feature. The
**deployment lock** (`deployment.lock.toml`) is that deployment captured as a
committed artifact; the **variation catalog** (`deployments/`) publishes
locks so a third party can adopt a deployment they have never run; `lobes
init --from-lock` restores one; `lobes doctor` reports `lock_drift` when the
deployed files stop matching.

This document is the deep reference. `lobes explain lock` is the brief
in-CLI version. The frame and build plan are
[`docs/specs/2026-08-29-deployment-lock-per-box.md`](specs/2026-08-29-deployment-lock-per-box.md)
and [`docs/plans/2026-08-29-deployment-lock-per-box.md`](plans/2026-08-29-deployment-lock-per-box.md).

> **Read the honesty section before you rely on any of this.** No real box
> has been captured, the catalog ships empty of real variations, and
> serve-after-restore has never been measured. See
> [What is not validated](#what-is-not-validated).

## The motivating incident, cited not remembered

On **2026-08-25**, preparing the cortex replica pool (#199), the Spark and
the Thor were found to be running genuinely different compose files for the
same role. The Spark's `docker-compose.yml` had been hand-edited to bake the
DSpark `--speculative-config` into `vllm-primary`'s command; the Thor's
happened to equal the packaged template. Only a live diff revealed it. The
acceptance transcript records both halves:

- [`docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt`](evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt),
  "Deploy record": *"the Spark's base compose is hand-edited with the DSpark
  speculative config and was NOT re-scaffolded — issue #214 proposes
  committing such files as a lock"* — the pool's own gateway passthrough
  lines had to be diverted into `docker-compose.override.yml` because
  re-scaffolding the Spark was not safe.
- The same transcript's fingerprint notes: *"PRIMARY_SPECULATIVE_CONFIG is
  unset on both boxes (the Spark's DSpark config is baked into its compose
  command, the Thor's MTP into the template default) — so the drafter
  difference is NOT visible in the fingerprint yet … #214 (commit the
  rendered compose as a lock) would close that gap."*

That is the before-state, in a committed file, with a date and an issue
number — not a claim in a frame.

### The three costs, each separately observable

| Cost | Observable today, by | Where |
|---|---|---|
| A hand-edited box cannot be re-scaffolded safely, so it cannot be shared or restored without re-applying every hand edit from memory | the transcript's own deploy record: the Spark was *not* re-scaffolded and the new lines went into `docker-compose.override.yml` instead | `docs/evidence/2026-08-25-accept-cortex-replica-pool-spark-thor.txt` |
| Evidence transcripts cite only `MODEL_GEAR_VERSION` plus a shape name, under-describing what actually ran | the live fingerprint reading `speculative_config: unknown` on **both** boxes while the two were in fact drafting differently | same transcript, fingerprint lines |
| Passthrough drift accumulates on a box operators are afraid to re-scaffold | `lobes doctor`'s pre-existing `gateway_passthrough` check (`lobes/cli/_commands/doctor.py`), which names gateway env keys the deployed compose does not pass through | `lobes doctor --json` |

The lock addresses the first two directly (a committed, restorable
deployment; a lock a transcript can cite) and makes the third safe to fix
(`lobes init --from-lock` restores a hand-edited box verbatim, so
re-scaffolding is no longer the only way back).

## Who reaches it, and by what mechanism

Every audience below reaches the artifact through something executable —
none is served only by prose. That is acceptance criterion 1 of the plan's
t12, and it is the reason each row names a command, a check or a job.

| Audience | Mechanism | Where |
|---|---|---|
| The fleet operator running lobes on more than one box | `lobes init --from-lock <dir>` restores a committed variation verbatim | `lobes/cli/_commands/init.py` |
| A third party adopting a variation they have never run | a repo checkout alone: `deployments/<id>/` carries the lock, the files and a `VARIATION.md`; `--from-lock` materialises it with no access to the measured box | `deployments/README.md`, `lobes/variation_catalog.py` |
| `lobes doctor` (machine consumer) | the `lock_drift` check diffs the lock's `[files]` digests and `[env]` values against the deployment | `lobes/cli/_commands/doctor.py` |
| CI (machine consumer) | the **required**, non-`continue-on-error` `secrets-scan` job runs `scripts/scan_deployment_secrets.py` over every committed deployment artifact | `.github/workflows/tests.yml` |

## What a lock records

`deployment.lock.toml` (`lobes/runtime/_lock.py`, `LOCK_FILENAME`,
`SCHEMA_VERSION = 1`) is TOML — the house format, and deliberately **not**
env-shaped:

```toml
schema_version = 1

[variation]
id = "spark"
profile = "spark"
shape = "spark-lobe"
lobes_version = "0.67.0"
evidence = "docs/evidence/....txt"   # or omitted entirely

[env]
PRIMARY_MODEL = "unsloth/Qwen3.8-27B-NVFP4"
PRIMARY_GPU_MEM_UTIL = "0.58"
# … the allowlisted rendered knobs, sorted

[files]
"docker-compose.yml" = "sha256:…"
"docker-compose.override.yml" = "sha256:…"
"Dockerfile.gateway" = "sha256:…"
```

### `[env]` is an allowlist, derived — not a redacted copy

`lock_keys()` is computed from the renderer's own tables, never from a
hand-written list of names:

1. the role-knob grammar — `ROLE_ENV_PREFIX` crossed with
   `_KNOB_ENV_SUFFIX` plus `MODEL` / `SERVED_NAME` / `FEASIBLE`;
2. every card-level `host_env` key any packaged built-in profile declares;
3. the activation keys an alternative-engine or opt-in lane renders
   (`LLAMA_CPP_ACTIVATION_ENV`, `OPT_IN_ACTIVATION_ENV`,
   `OPT_IN_CORE_ACTIVATION_ENV`).

`build_lock()` takes a whole deployed `.env` — secrets included — and passes
it through `allowlist_env()`. A credential is not a `RoleProfile` field, so it
can never enter the set without someone deliberately declaring it a rendered
knob. A denylist would silently ship the next secret key someone adds; this
cannot. `write_lock()` re-checks at the last moment (`_assert_secret_free`),
so even a hand-constructed `DeploymentLock` is refused before bytes reach a
file.

### Two exclusions narrow the allowlist (deviation `d1`)

The spec described a *pure* allowlist. What shipped is the allowlist **minus
two exclusion rules** (`is_excluded()`), recorded as deviation `d1`
(**proposed, not approved**):

- **`COMPOSE_PROFILES`** — the renderer does emit it, but it is
  operator-typed state (`_compose.MERGE_ONLY_FILES` names it beside the
  bearer key and the Hub token): a box's opt-in gears (`muse`, `worker`,
  `minor`) are declared there, and a lock re-stating a rendered value would
  fight them.
- **any key ending `_URL`** — `*_BASE_URL` / `PRIMARY_URL` are wiring facts.
  The renderer only ever writes a compose-internal DNS name
  (`http://vllm-worker:8000`), but the same key is retargetable by hand at
  another box, which would put an internal origin into a committed file. A
  restore re-renders the wire from the shape anyway.

Both rules can only ever *remove* keys — nothing enters the lock by failing
to match them.

### `[files]` is a digest table, not content

`file_digest()` produces `"sha256:<hex>"`. The table maps a plain filename to
its digest; it never carries file content. Two consumers read it, and they
want slightly different things from it — see the `.env` note under
[Drift](#drift-doctor-lock_drift-and-the-switch-warning).

## Restoring: `lobes init --from-lock`

```bash
lobes init --from-lock deployments/spark            # dry run (default)
lobes init --from-lock deployments/spark --apply    # writes
lobes init --from-lock deployments/spark/deployment.lock.toml --apply
```

`--from-lock` is a **fourth thing entirely: a SOURCE, not another input to
the renderer.** `lobes init`'s other three axes (topology `--single` /
`--audio`, card `--profile`, shape `--shape`) all feed
`lobes/profiles/render.py`; `--from-lock` bypasses that path completely and
is refused alongside any of them. That bypass is what makes a restore
byte-identical to what the box *ran*, hand edits included, rather than to
what the renderer would produce today
(`tests/test_init_from_lock.py::test_restore_never_consults_resolution`
patches resolution to raise and the restore still succeeds).

Everything that can refuse does so **before the first byte is written**, and
the dry run runs the identical checks:

- the lock parses and its `schema_version` is understood;
- `[files]` is non-empty and every name is a plain filename — no separator,
  no `..`, and nothing in the `.env` family;
- every named file is present in the variation folder and its bytes match the
  recorded digest;
- the machine type agrees (below);
- the buildability preflight runs (below).

### The machine-type guard

Bypassing resolution also bypasses `_sync_gpu_overrides`, the card-driven
correction that decides whether a deployment asks for the GPU the modern
(`deploy.resources`) or the legacy (`runtime: nvidia`) way. Restoring a
csv-mode variation onto a devices-mode board — or the reverse — would
silently reproduce exactly the bug those overlays exist to fix. So the lock's
declared variation id is checked against what `lobes/runtime/_detect.py` +
`lobes/variation.py` resolve on this box, and a mismatch **refuses**:

```text
error: refusing to restore a variation captured on another machine type:
this box detects as 'thor' but the lock declares variation 'spark'
```

An **UNKNOWN** card is a mismatch, not a pass — "we could not tell" is not
evidence that the lock fits. The override is its own flag,
`--allow-variation-mismatch`, never `--force`: restoring onto the wrong
machine type is a different decision from clobbering a file, and conflating
them would let one be taken while meaning the other. With the flag, a warning
names what is being accepted (*"the restored GPU-access overlays are the
LOCK's, not this card's"*) and the restore proceeds.

Variation identity is **machine type or setup, never hostname**
(`lobes/variation.py`): two physically different Sparks resolve to the same
id, and an unrecognised card resolves to an explicit unknown rather than a
guessed nearest match.

### What a restore writes, removes, and never touches

- **Writes**, verbatim: every file the lock's `[files]` table names.
- **Removes**: the *generated* overlays this lock does not name —
  `docker-compose.shape.yml`, `docker-compose.gpu.yml`,
  `docker-compose.gpu-audio.yml` (`RESTORE_SYNCED_FILES`). This preserves the
  remove-on-mismatch behaviour of `_sync_shape_override` /
  `_sync_gpu_overrides` across a lock round trip. Nothing outside that tuple
  is ever deleted — an operator's own `docker-compose.override.yml` is not
  lobes' to remove.
- **Never rewrites `.env`.** `.env` is the sole member of
  `_compose.MERGE_ONLY_FILES`. A restore *appends* the lock's rendered knobs
  where the `.env` lacks them and leaves every existing line byte-identical,
  in place and in order (`_merge_lock_env`, deliberately not `_env.set_env`,
  which rewrites the whole file even for a pure append). On a box with no
  `.env` at all, one is created holding only the lock's knobs, under a header
  spelling out that no credential is restorable from a lock.

The restore's closing line says what still has to happen: *"next: supply this
box's secrets (`GATEWAY_API_KEY`, `HF_TOKEN`, any `*_PEER_API_KEY`), then
'lobes fleet up --apply'"*.

### Does a restored box serve the same model at the same knobs?

**The mechanism guarantees the inputs; nobody has measured the output.**

What is proven, by test rather than by assertion: the restored
compose/override/Dockerfile set is byte-identical to the captured one
(`test_restore_is_byte_identical_to_the_captured_box`), an existing `.env`
comes through byte-identical
(`test_restore_leaves_an_existing_env_byte_identical`,
`test_restore_never_rewrites_an_existing_env_line`), and the locked knob
values land in `.env` unchanged. Since the served model and its knobs are
exactly a function of those files plus `.env`, the same inputs are present
after a restore as before it.

What is **not** proven: no box has been restored from a lock and then served.
There is no acceptance transcript for serve-after-restore, and per
CLAUDE.md's #108 rule nothing here may call that validated. See
[What is not validated](#what-is-not-validated).

## Drift: `doctor lock_drift` and the `switch` warning

`lobes doctor` gains one check, `lock_drift`
(`lobes/cli/_commands/doctor.py`):

- **No lock present → no finding at all.** Absence of a lock is not drift; a
  deployment that never adopted the practice behaves exactly as before.
- **Match → `passed: true`, severity `info`.**
- **Divergence → `passed: false`, severity `warn`,** naming the *specific*
  differing files and locked keys — never merely "drift exists". A tracked
  file that has vanished is named `<name> (missing)`.

`lock_drift` is **read-only**. `doctor --fix` never calls it, so the
missing-only heal lane's convention — write absent files and append absent
keys, never rewrite an existing `.env` line — is untouched by this check's
existence (`test_fix_apply_ignores_lock_drift_and_never_rewrites_an_existing_env_line`).

The env half re-derives the allowlisted subset of the *currently deployed*
`.env` through the same `allowlist_env()` the writer uses, so the check can
never flag a secret key — only the rendered knobs the lock is permitted to
carry.

### Staleness from `lobes switch` — and why it is caught by a digest (deviation `d4`)

The spec's claim c33 said `lobes switch` writes the key family the lock
captures, so a first-class verb — not only a hand edit — makes a lock stale.
**That claim is wrong**, and deviation `d4` (**proposed, not approved**)
records it: `switch` writes only the legacy `VLLM_*` keys, which never
intersect `lock_keys()` (built from `ROLE_ENV_PREFIX`, i.e. `PRIMARY_*` /
`MULTIMODAL_*` / …).

The conclusion survives the correction, by a different mechanism. A lock may
record `.env`'s **own digest** in `[files]` — a hash, never its content, so
it stays outside the secret-free contract — and `_lock_file_diffs` diffs any
tracked name identically. A `switch --apply` therefore shows up as `.env`
differing, named specifically
(`test_switch_apply_makes_the_env_tracked_lock_stale_and_doctor_reports_it`).

`lobes switch` itself warns before writing, whenever a lock is present,
independent of which keys this particular switch touches:

```text
a committed deployment.lock.toml is present — this switch writes to .env,
so re-capture and commit deployment.lock.toml afterwards (or run 'lobes
doctor' to see what now differs) so the lock keeps describing this box
```

> **A lock that tracks `.env`'s digest is not restorable.**
> `--from-lock` refuses any `[files]` name in the `.env` family
> (`_check_restorable_name`), so a lock built for `.env`-digest drift
> detection fails the restore path outright, before anything is written. The
> two uses are separable today only by capturing different `[files]` tables:
> track `.env` in a box-local lock for drift, and omit it from a lock
> published for adoption. Nothing in the tree reconciles them.

## The variation catalog — `deployments/`

The repo publishes **all** docker variations, not only the boxes this
operator happens to run. A chooser picks one and adopts it; a chooser needs
the files to run it *and* an honest statement of what running it produced.
Layout, contract and validator live in `lobes/variation_catalog.py`, with the
front matter in [`deployments/README.md`](../deployments/README.md).

```text
deployments/
  README.md                     # the catalog's front matter
  VARIATION.template.md         # copy this — not itself a variation
  <variation-id>[__<shape>]/
    deployment.lock.toml
    VARIATION.md
    docker-compose.yml          # …and every other file the lock names
    docker-compose.override.yml
    Dockerfile.gateway
```

The directory **name** is the variation id, optionally suffixed
`__<shape>`; both halves are cross-checked against the lock's own `variation`
/ `shape` fields, so a directory cannot quietly describe a different box than
its lock does.

`validate_variation()` reports a *list* of problems (a chooser wants them all
at once) covering: the lock loads; `[files]` is non-empty; every named file is
present with matching bytes; nothing unrecorded sits beside them; and the info
file makes an honest, resolvable evidence claim.

### The info-file contract — evidence cited, or absent out loud

A `VARIATION.md`'s `## Measured result` section does **exactly one** of two
things:

1. **cites** one or more existing `docs/evidence/` transcripts by path — it
   cites rather than restates the numbers, because the transcript is the
   measurement and a second copy of the figures would drift from it; every
   cited path must resolve, and the lock's own `evidence` field must be among
   them; or
2. **states, verbatim, `No measured result.`** — and the lock's `evidence`
   field is then absent.

Never both, and never neither. A blank section is a **finding**, not a
default. This exists because CLAUDE.md's #108 rule keeps a shape DECLARED
until an acceptance transcript lands: all four Orin shapes and `thor-muse`
have none today, so most of this catalog will honestly read "no measured
result", and publishing it must not manufacture the appearance of evidence.

## Secrets never enter the catalog

Four mechanisms, layered:

1. **A positional `.gitignore` rule** (`.gitignore`): any name *ending* in
   `.env` is ignored by construction — `.env`, `.cf-tunnel.env`,
   `.secrets.env`, and every future secret dotfile, with no new line to
   forget. A name that *starts* with `.env.` (`.env.example`, `.env.sample`)
   stays tracked. The `tests/goldens/` tree commits real `*.env` fixtures on
   purpose, so a `!tests/goldens/**/*.env` negation keeps a newly generated
   golden stageable — proven by
   `tests/test_gitignore_convention.py`, including a test that fails if the
   negation is removed.
2. **The lock's allowlist** (above) — the generated half is secret-free by
   construction.
3. **`scripts/scan_deployment_secrets.py`** — the mechanical gate over the
   half the allowlist does *not* protect. `docker-compose.override.yml` is
   the one file lobes never writes, never templates and cannot vouch for, and
   the Spark's baked `--speculative-config` proves operators do hand-edit
   these files. The scanner's globs name the lock and every verbatim-committed
   compose/override/Dockerfile under `deployments/` **explicitly** — not "the
   repository's default file set" — and it flags known deployment-secret key
   names (`GATEWAY_API_KEY`, `CULTURE_VLLM_API_KEY`, `HF_TOKEN`, and every
   `*_PEER_API_KEY(S)` / `*_PEER_ORIGIN(S)` suffix, so a future role is
   covered without enumerating roles) carrying a non-empty, non-template
   value. It runs in CI as the **required** `secrets-scan` job, over both
   `deployments/` and `tests/fixtures/`.
4. **A secret file family beside `.env`.** Every fleet service that reads
   `.env` also reads `.secrets.env`, as a second long-form
   `env_file: [{path: …, required: false}]` entry — so a deployment with no
   such file behaves byte-identically (`tests/test_fleet_secrets_env_file.py`).
   With `scripts/gen-api-key.py` generating the bearer key rather than
   printing it, a fresh box reaches a serving state without the operator
   typing a secret value by hand.

Recovery from a leak — every place a copy of an inbound key lives, the drill,
and why `git rm` is not enough — is
[`docs/secret-rotation.md`](secret-rotation.md).

The wheel never ships any of this: `pyproject.toml`'s
`[tool.hatch.build.targets.wheel]` declares `packages = ["lobes"]`, and
`tests/test_wheel_excludes_deployments.py` builds a wheel and asserts its
namelist carries no `deployments/` path (plus a companion test proving a
widened `packages=` *would* ship it).

## The buildability preflight

`Dockerfile.gateway` installs the gateway as `pip install
lobes-cli==${MODEL_GEAR_VERSION}` — from PyPI for a release, from a TestPyPI
dev index for a `.devN` PR build. A variation captured at a dev version
references an artifact that may not outlive the PR, so the catalog's promise
("a third party can materialise a variation they never ran") can fail at
**build** time, not restore time.

`lobes/runtime/_buildability.py` classifies the pin, and
`lobes init --from-lock` runs it as a preflight (`_guard_buildability`),
before anything is written:

| `risk` | Meaning | What `--from-lock` does |
|---|---|---|
| `unversioned` | no `lobes_version` recorded | warns: `docker compose build gateway` will resolve an empty pin unless `.env` supplies one |
| `ephemeral_dev` | PEP 440 dev-release shaped (`0.67.0.dev12`) | warns: published to TestPyPI by a PR, may no longer be installable |
| `released` | ordinary release shape | silent |

**This preflight is offline and warn-only, deliberately, and it cannot prove
a wheel uninstallable.** A restore must not depend on network reachability,
and offline nothing here can distinguish "gone" from "not checked". The
module carries a second, *injectable* layer for that — `IndexQuery`, with
`default_pypi_index_query` as the one function that touches the network and
`assert_buildable()` as the raising path on a definitive `installable is
False` — but **no caller wires it today**. Deviation `d3`
(**proposed, not approved**) records that t10's acceptance criterion "the
check fails early, not deep inside a docker build" is therefore only half
delivered: the *warning* fires early; nothing *raises* early, because nothing
asks an index.

## Success signals, and how each fails on a broken input

All three are mechanical — a test, a doctor check, or a CI job — and none
requires a human judging output. Each row's second column is the deliberately
broken input that makes it fail; that is what distinguishes a signal from a
formality.

| Signal | Passes on | Fails on |
|---|---|---|
| A restore is byte-identical, `.env` untouched | `test_restore_is_byte_identical_to_the_captured_box`, `test_restore_leaves_an_existing_env_byte_identical` | a tampered variation file (`test_a_digest_mismatch_is_refused_before_any_write`), a missing file (`test_a_missing_committed_file_is_refused_before_any_write`), an unsafe `[files]` name (`test_a_lock_naming_an_unsafe_file_is_refused`), an empty `[files]` table (`test_a_lock_naming_no_files_is_refused`), the wrong machine type (`test_restore_refuses_when_the_detected_card_differs`, and `test_an_unknown_card_is_a_mismatch_not_a_pass`) |
| `doctor` reports `lock_drift` naming the differing files | `test_lock_drift_passes_when_nothing_diverges` | a hand-edited compose file (`test_lock_drift_names_the_specific_differing_file`), a hand-edited knob (`test_lock_drift_names_the_specific_differing_locked_key`), a deleted tracked file (`test_lock_drift_names_a_missing_tracked_file`), a real `switch --apply` (`test_switch_apply_makes_the_env_tracked_lock_stale_and_doctor_reports_it`) |
| CI fails a PR committing a credential- or origin-shaped value | `test_clean_tree_passes`, `test_real_repo_tree_is_clean` | a token planted in an override (`test_planted_token_in_override_fails`), in the lock itself (`test_planted_token_in_lock_fails`), in a Dockerfile (`test_planted_hf_token_in_dockerfile_fails`), a peer key (`test_planted_peer_api_key_fails`), a peer origin (`test_peer_origin_is_treated_as_sensitive`), or in a catalog fixture (`test_secret_scanner_would_catch_a_planted_token_in_a_variation`) |

Two supporting gates fail the same way: the lock writer refuses to render a
lock carrying a non-allowlisted key even when hand-constructed
(`test_write_lock_refuses_to_write_a_lock_carrying_a_secret`), and the
gitignore negation test fails if the goldens exception is removed
(`test_negation_removal_breaks_goldens_staging`).

## What is not validated

Per CLAUDE.md's #108 rule, nothing below may be described as validated, and
this section exists so no reader has to infer it.

- **No real box has been captured.** `deployments/` ships a README and a
  template and **zero variations**. Every catalog behaviour above is
  exercised against clearly-labelled fixtures under
  `tests/fixtures/deployments/`. No Spark, Thor or Orin variation exists;
  do not read this page as evidence that one does.
- **There is no capture verb.** `capture_lock()` / `build_lock()` /
  `write_lock()` are a library
  (`lobes/runtime/_lock.py`) with no CLI caller anywhere in `lobes/cli/`.
  Every "re-capture and commit the lock" instruction — in `doctor`'s
  remediation, in `switch`'s warning, in `deployments/README.md` — currently
  means *call the library*, not *run a command*. Producing the first real
  variation requires that gap to close.
- **Serve-after-restore is unmeasured.** "A restored box serves the same
  model at the same knobs as before the restore" is an argument from
  byte-identical inputs plus a merge-only `.env`, both test-proven — not a
  measurement. No box has been restored and served, and no acceptance
  transcript exists.
- **A lock-restored FRESH box is not yet servable** (deviation `d2`,
  **proposed / needs-follow-up**). The fleet compose bind-mounts
  `mg-logwrap.sh` (the durable-log entrypoint, issue #50) and
  `qwen3_thinking_tool_parser.py` (the cortex tool-parser plugin). Both are
  packaged **scaffold** files, not compose files or Dockerfiles, so a
  variation whose `[files]` table names only compose + Dockerfiles restores an
  *incomplete* deployment. `--from-lock` accepts any plain non-`.env`
  filename, so naming them in `[files]` is the obvious workaround — but no
  capture path does so today and no test covers it.
- **The buildability preflight cannot prove a wheel uninstallable** — it is
  offline and warn-only; the raising path needs an opt-in index query that is
  not wired (deviation `d3`).
- **Four deviations were recorded during implementation and are `proposed`,
  not approved** (`devague deviate --list`): `d1` (the lock is an allowlist
  minus two exclusion rules, not the pure allowlist the spec claims), `d2`
  (fresh-box restore incomplete), `d3` (the buildability guard's raising path
  has no caller), `d4` (spec claim c33 is wrong; staleness is caught via an
  `.env` digest, not the allowlist). `d1` and `d4` are marked *acceptable*;
  `d2` and `d3` are marked *needs-follow-up*.
- **A lock cannot both track `.env` for drift and be restorable** — see the
  note under [Drift](#drift-doctor-lock_drift-and-the-switch-warning).
- **`lock_drift` can only check the LOCAL variation.** A catalog of
  variations for boxes nobody here runs has nothing to diff against, so most
  published variations are unverifiable by the mechanism that keeps the local
  one honest. Nothing forces a lock to be re-committed after a hand edit
  either; whether `lock_drift` is a sufficient forcing function is an open
  question the frame parked, not one this implementation answered.
- **There is no trust model for adopting a third party's variation.** A
  variation names build contexts, bind mounts and image refs, so adopting one
  is closer to running a script than reading a doc. The frame's challenge
  pass flagged this surface as *not examined*; it still is.

## See also

- `lobes explain lock` — the in-CLI brief
- [`deployments/README.md`](../deployments/README.md) — the catalog front
  matter and the "adding a variation" procedure
- [`docs/secret-rotation.md`](secret-rotation.md) — leak recovery, every copy
  of an inbound key, the history-rewrite step
- [`docs/deployment-shapes.md`](deployment-shapes.md) — the orthogonal axis:
  which roles a box hosts
- [`docs/machine-profiles.md`](machine-profiles.md) — the other orthogonal
  axis: how a role is tuned on a card
- `lobes/runtime/_lock.py`, `lobes/variation.py`,
  `lobes/variation_catalog.py`, `lobes/runtime/_buildability.py`,
  `scripts/scan_deployment_secrets.py` — the implementation
