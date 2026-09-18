# Golden rendered artifacts per shipped profile (t13)

This directory holds the **byte-for-byte** golden files
`tests/test_profile_goldens.py` diffs against on every run:

* `spark.env`, `thor.env` — one per packaged built-in profile (see
  `lobes.profiles.loader.builtin_names()`), the sorted `KEY=VALUE` projection
  of `profile_env(resolve_profile(<name>))` (`lobes/profiles/render.py`,
  `lobes/profiles/loader.py`). A future built-in profile picks up its own
  golden automatically (the test parametrizes over `builtin_names()`), but the
  golden file itself must still be generated and committed.
* `template-defaults.env` — the substitution surface of
  `lobes/templates/fleet/docker-compose.yml` (extracted by
  `regen.extract_template_defaults`): every var with an explicit `${VAR:-default}`,
  written `VAR=default`, plus every CONDITIONAL `${VAR:+alternate}` slot,
  written `VAR:+alternate`. This is the "GB10 mostly runs on template
  defaults" surface — it catches a template edit that silently changes what an
  UNRESOLVED profile knob renders to, which a profile-only golden can't see —
  and, since the `:+` half landed, an edit to the flag a conditional knob
  composes (`WORKER_MOE_BACKEND` renders nothing when unset, so nothing else
  in this directory would move if its flag spelling changed). The dash-only
  `${VAR-}` slots (`*_SPECULATIVE_CONFIG`, the worker boolean toggles) are
  deliberately NOT here: they have no default to drift.

* `overlays/audio-he-defaults.env` — the same substitution surface for the
  opt-in **Hebrew audio overlay**, `lobes/templates/fleet/docker-compose.audio-he.yml`
  (hebrew-realtime t15). Language is an overlay choice, not a shape and not a
  variation, so it gets an `overlays/` golden of its own rather than a
  per-profile or per-shape one. It lives in a subdirectory because
  `test_golden_file_set_matches_builtin_profiles` asserts the top-level `*.env`
  set equals the built-in **profile** set — an overlay is not a profile.
  Editing the Hebrew overlay must leave every other golden here
  byte-identical, and vice versa: the English overlay
  (`docker-compose.audio.yml`) has no golden at all, so a change that moves
  *both* audio files is exactly the kind of diff to read twice.

Regenerate all of them with:

```sh
uv run python tests/goldens/regen.py
```

## Why this exists

Rendering a profile is a pure function of `(profile, template)` — no host
state, no GPU, no subprocess. That means these goldens run identically on any
dev box, including GPU-less CI (`tests/test_profile_goldens.py` also has a
purity meta-test that renders twice and asserts identical output, and asserts
`resolve_profile`/`profile_env` complete with `HOME` pointed at an empty temp
dir, a trimmed `os.environ`, and `subprocess.run`/`Popen` patched to raise).

**Editing the Thor bundle (`lobes/profiles/builtin/thor.toml`,
`lobes/machines/thor.py`, or the shared `SM_110` trait in
`lobes/machines/_traits.py`) must leave `spark.env` and
`template-defaults.env` byte-identical — and vice versa.** Each shipped
profile has its own golden precisely so a change scoped to one machine can't
silently move another's rendering. `tests/test_profile_goldens.py` also
directly asserts `spark.env` never contains Thor's four validated sm_110
divergences (`TRITON_ATTN`, `--enforce-eager`, `PRIMARY_KV_CACHE_DTYPE=auto`).

If a change is **deliberately** cross-machine (e.g. a compose template edit
that changes a default both profiles inherit), update the OTHER golden(s) in
the **same commit** — the diff in that commit is the review surface a
reviewer (human or CI) checks against the stated intent. A PR that moves
`thor.env` and *also* moves `spark.env` or `template-defaults.env` without
saying why is exactly the failure this suite is built to surface.
