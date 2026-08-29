# Secret rotation and revocation

This is a recovery procedure, not a preventive one. `.gitignore`'s bare
`.env` line (and the explicit `.cf-tunnel.env` line beside it) keep the
deployment secret files unstaged; `scripts/gen-api-key.py` rotates the
gateway bearer key **locally**. Neither answers the question this doc
answers: **a credential leaked (committed, pasted into a chat, logged) — what
do I do, in what order, and how do I know each step actually worked?**

Two facts make the answer bigger than "generate a new key":

1. **The blast radius is bigger than one box.** Under proxy-lobes and the
   cortex replica pool, the credential model is pairwise-by-copy, not
   per-pairing: each box has exactly ONE inbound key
   (`GATEWAY_API_KEY`, with `CULTURE_VLLM_API_KEY` as a fallback source —
   see `docs/gateway-fleet.md#auth-opt-in-bearer-gate`), and every peer that
   dials it holds a **copy** of that same value as its own outbound
   `<PREFIX>_PEER_API_KEY` (singular peer) or a positional slot in
   `<PREFIX>_PEER_API_KEYS` (replica-pool plural peer). Rotating a leaked key
   is therefore not one edit — it is one edit on the box that owns the key,
   plus one edit on every peer holding a copy of it.
2. **`git rm` does not remove a committed secret.** Deleting the file (or the
   line) in a new commit only stops it appearing in the *current* checkout —
   the value is still readable in every commit before that one, in
   `git log -p`, in any clone or fork already made, and in GitHub's own
   history views. Recovering from a committed secret requires rewriting
   history, not just editing it forward, and the leaked value has to be
   treated as burned regardless of whether the rewrite ever lands everywhere.

If you take one thing from this page: **rotating the value always comes
before, or at worst alongside, cleaning up where it was exposed.** A key
still valid after a leak is still a leak, no matter how thoroughly the
commit that exposed it gets scrubbed.

## Every place a copy of a key lives

There is exactly one *inbound* credential per box, and it is copied outward
to every peer that talks to that box. Before touching anything, work out
which of these apply to the box whose key leaked (grep the deployment `.env`
for the names below — see "Find every copy" further down for the exact
commands).

**The inbound key, on the box that owns it (one of these two):**

- `GATEWAY_API_KEY` — the explicit, gateway-scoped bearer key.
- `CULTURE_VLLM_API_KEY` — a fallback source for the same inbound key
  (`GATEWAY_API_KEY` wins if both are set; the first non-blank of the two
  gates every request). `scripts/gen-api-key.py` writes into this name.

**Copies on every peer that dials this box, one pair of vars per role prefix
this box hosts and the peer proxies/pools:**

- `<PREFIX>_PEER_API_KEY` — proxy-lobes singular form (one peer per dropped
  role).
- `<PREFIX>_PEER_API_KEYS` — replica-pool plural form (issue #199),
  **positional** against that role's `<PREFIX>_PEER_ORIGINS` list (index *i*
  is the outbound key for peer *i*; an empty slot — `k1,,k3` — is legal and
  means that peer has no inbound gate).

`<PREFIX>` is one of the ten role prefixes the gateway's `FEASIBLE_ENV` /
peer-channel machinery knows about (`lobes/gateway/_config.py`):
`PRIMARY`, `MULTIMODAL`, `MUSE`, `WORKER`, `ASSOCIATE`, `HAND`, `EMBED`,
`RERANK`, `STT`, `TTS`. Not every prefix is wired on every box — only the
ones a peer actually proxies or pools toward this box carry a copy. See
`docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in` and
`docs/gateway-fleet.md#replica-pools-one-lobe-n-replicas-opt-in-cortex-validated-only`
for the mechanism these variables drive.

**What is NOT a copy of the leaked key:** a caller's own `Authorization`
header is stripped before every proxy/pool forward and never reaches a peer
(`docs/gateway-fleet.md`, "pairwise credential model"), so a caller-side
credential leak does not, by itself, imply any `<PREFIX>_PEER_API_KEY*` was
exposed. Only the box's own inbound key, and copies of it operators typed
into peers, are in scope for this drill.

**Everything below is a live-box operation — commands you run against a
reachable deployment.** This page documents the procedure; it has not been
exercised end-to-end against a live fleet box as part of writing it. Treat
each command as reviewed-correct against the current tree, not as a run
transcript.

## Find every copy

Before rotating, enumerate what actually needs changing. On the box that
owns the leaked key:

```bash
grep -n '^GATEWAY_API_KEY=\|^CULTURE_VLLM_API_KEY=' "$LOBES_DIR/.env"   # or ~/.lobes/.env
```

On every box you believe proxies or pools to it (you may need to check each
peer box individually — there is no fleet-wide inventory command today):

```bash
grep -n '_PEER_API_KEY=\|_PEER_API_KEYS=' "$LOBES_DIR/.env"
```

Cross-reference the non-blank `<PREFIX>_PEER_ORIGIN`/`<PREFIX>_PEER_ORIGINS`
values against the leaked box's own address to confirm which peer entries
are actually copies of *this* box's key, as opposed to some other box's.

## The leaked-key drill

Run these in order. Each step names its own verification — do not move to
the next step until the current one's check passes.

### 1. Rotate the inbound key on the box that owns it

```bash
python3 scripts/gen-api-key.py --force --dir "$LOBES_DIR"
```

This overwrites `CULTURE_VLLM_API_KEY` in place (never rewriting any other
line — see the script's own `_write_key`), sets the `.env` file mode to
`0o600`, and does **not** print the new value (pass `--show` only if you are
about to paste it straight into a peer's `.env` in the same terminal
session — never into chat, a ticket, or a commit).

If the leaked key was set as `GATEWAY_API_KEY` rather than
`CULTURE_VLLM_API_KEY` (the script only writes the latter), also rotate
`GATEWAY_API_KEY` directly — generate a fresh value with the stdlib
equivalent the script uses and write it into `.env` by hand, or clear
`GATEWAY_API_KEY` so `CULTURE_VLLM_API_KEY` becomes the effective source
(`GATEWAY_API_KEY` wins over `CULTURE_VLLM_API_KEY` when both are set — see
`docs/gateway-fleet.md#auth-opt-in-bearer-gate`, so a blank `GATEWAY_API_KEY`
plus a rotated `CULTURE_VLLM_API_KEY` is a valid rotation, not a half-done
one).

**Verify:** read the new value back privately (`grep` the `.env` file
directly — the script deliberately never echoes it) and confirm it differs
from the leaked value.

### 2. Restart the gateway to enforce the new key

```bash
lobes serve --apply   # or: lobes fleet up --apply, for a fleet deployment
```

The bearer gate reads the key from the environment at container start; a
rotated `.env` value has no effect until the gateway process is restarted.

**Verify:**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/v1/models \
  -H "Authorization: Bearer <the-old-leaked-key>"
```

expect `401`. Then repeat with the new key and expect `200` (or the
deployment's actual bound port/host).

### 3. Update every peer holding a copy

For each peer box identified in "Find every copy" above, edit its `.env` in
place — replace the stale `<PREFIX>_PEER_API_KEY` value, or the matching
positional slot in `<PREFIX>_PEER_API_KEYS`, with the box's new key from
step 1 — then restart that peer's gateway the same way as step 2.

**Verify, from the peer:**

```bash
curl -s http://<rotated-box-origin>/capabilities \
  -H "Authorization: Bearer <peer's-new-outbound-key>" | head -c 200
```

A forwarded request should succeed (`X-Lobes-Proxied-By` on a proxy-lobes
route, or `X-Lobes-Served-By`/`X-Lobes-Proxied-By` with
`X-Lobes-Route-Reason` on a pooled route — see
`docs/gateway-fleet.md#proxy-lobes-the-third-lobe-state-opt-in` and the
"Replica pools" section for the exact headers). A peer left on the stale key
gets `401` from the rotated box — that 401 is the signal you missed a peer;
go back to "Find every copy" and check it again.

### 4. Confirm nothing else still authenticates with the old value

Repeat step 1's curl probe (old key → expect `401`) from every box in the
mesh that was ever configured to reach this one, not only the peers you
found in step 3 — a stale key baked into a script, a CI secret, or an
operator's shell history is still a live credential until it fails.

### 5. If the leak was a committed file: treat the value as burned regardless

`git rm` (or deleting the offending line and committing that) removes the
secret from the tip of the branch only. It is still present in:

- every commit before the removal, reachable via `git log -p -- <path>` or
  `git show <sha>:<path>`;
- any fork or clone already made before the removal lands;
- GitHub's own commit/blame history views, which serve old commits directly.

Steps 1–4 above (rotate, restart, verify) must happen regardless of whether
or when history gets rewritten — a still-valid key sitting in old commits is
exactly as usable by an attacker as one sitting in the current file. Do not
wait for the history rewrite to finish before rotating; do not treat the
rewrite as a substitute for rotating.

To actually remove the value from history (needed once the leaked commit has
been pushed anywhere, not only to satisfy internal hygiene):

```bash
# git filter-repo is not vendored in this repo; install it first
# (pipx install git-filter-repo, or the OS package), then from a
# fresh clone (filter-repo refuses to run in-place on your working clone):
git filter-repo --path <leaked-file> --invert-paths
```

or, for a value embedded inline rather than a whole file worth removing, use
`git filter-repo`'s `--replace-text` against a file listing the literal
leaked string. After the rewrite:

- force-push the rewritten history to every remote that held the leaked
  commit (`git push --force` to `origin`, coordinating with anyone else who
  has a clone — a rewrite invalidates every existing clone's history);
- ask GitHub support to purge cached views of the removed commit (GitHub
  documents this at
  <https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository>)
  since a rewrite alone does not clear GitHub's own commit cache or anything
  already indexed by a search engine or a fork nobody told you about.

**Verify:** `git log --all -p -- <path>` (or `git log --all -S '<leaked
value>'`) on a fresh clone taken after the force-push returns nothing.

## What this drill does not cover

- **Non-gateway secrets** (`HF_TOKEN`, and any credential a future
  `deployment.lock.toml`/`deployments/<box>/` artifact might one day need to
  exclude) are out of scope here — `HF_TOKEN` is a Hugging Face token, and
  its own provider-side revocation happens on huggingface.co, not in this
  fleet.
- This page describes today's `.env`-based deployment. A per-box committed
  lock (`docs/plans/2026-08-29-deployment-lock-per-box.md`) is designed so
  that a committed artifact never contains a secret in the first place —
  the lock is an allowlist of non-secret rendered keys, not a redacted copy
  of `.env` — which is why this drill does not need a separate lock-specific
  rotation path: if the allowlist design holds, there is nothing secret in a
  committed lock to rotate.
