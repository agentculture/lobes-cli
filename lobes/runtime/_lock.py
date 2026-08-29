"""The committed deployment lock — ``deployment.lock.toml`` (t6).

A box's deployment is a *committed* artifact: the compose files, overrides and
Dockerfiles it actually runs are versioned in the repo under
``deployments/<variation>/``, alongside this lock, which records the rendered
knob state the box serves at. ``lobes init --from-lock`` (t7) restores a box
from it and ``lobes doctor`` (t8) reports ``lock_drift`` when the deployed
files no longer match.

**The safety argument this module embodies.** The lock is committed, so it must
be secret-free *by construction*, not by redaction. Its contents are therefore
an **ALLOWLIST of rendered keys** (:func:`lock_keys`), never a
denylist-filtered copy of the deployed ``.env``. Three facts make the allowlist
provably safe:

* :func:`lobes.profiles.render.profile_env` renders only knob keys —
  ``<ROLE>_{MODEL, SERVED_NAME, GPU_MEM_UTIL, MAX_MODEL_LEN, QUANTIZATION,
  KV_CACHE_DTYPE, ATTENTION_BACKEND, ENFORCE_EAGER, MAX_NUM_SEQS,
  HF_OVERRIDES, ALLOW_LONG_MAX_MODEL_LEN, SPECULATIVE_CONFIG, FEASIBLE}`` —
  plus a card's declared ``host_env`` table and a non-vLLM lane's activation
  URL. Nothing secret is in that set.
* 36 of the 37 committed goldens under ``tests/goldens/`` contain no
  ``API_KEY`` / ``HF_TOKEN`` / ``PEER_ORIGIN`` at all. The single hit,
  ``tests/goldens/template-defaults.env``, lists those key NAMES with empty
  values because it is the template's ``${VAR}`` default surface, not a render.
* By contrast :data:`lobes.runtime._compose.MERGE_ONLY_FILES`'s docstring names
  the operator-typed state a template can never regenerate — the inbound bearer
  key, every ``*_PEER_*``, ``COMPOSE_PROFILES``, the Hub token. None of it may
  ever enter the lock, and peer origins are internal information by operator
  decision, so they are excluded too.

A denylist silently ships the next secret key someone adds; an allowlist
structurally cannot. That is why nothing here enumerates a forbidden name: the
key set is DERIVED from the renderer's own tables, so a knob added to
:mod:`lobes.profiles.render` is picked up automatically while an unrecognised
key — secret-shaped or not — is dropped without anyone having to notice it.

**Why not an env-shaped filename.** The repo's positional gitignore convention
ignores a ``.env`` SUFFIX and allows a ``.env.`` PREFIX. An env-shaped
committed file is exactly where someone would paste a real ``.env`` and blank a
few lines by hand, so the lock is deliberately named as neither.
``deployment.lock.toml`` also matches the house format
(``lobes/profiles/builtin/*.toml``, ``lobes/profiles/builtin_shapes/*.toml``).

**Identity is a parameter, never detected here.** The variation id (machine
type, never a hostname) is resolved by t2 and passed in by the caller. This
module does no detection of any kind, which is what keeps it a pure function of
its inputs.

Stdlib only. Python's :mod:`tomllib` reads TOML but cannot write it, so the
serializer below emits the small, fully-quoted subset this schema needs
(:func:`lock_toml`) and :func:`load_lock` reads it back with :mod:`tomllib`.
"""

from __future__ import annotations

import hashlib
import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import (  # the allowlist is DERIVED from these tables
    _KNOB_ENV_SUFFIX,
    LLAMA_CPP_ACTIVATION_ENV,
    ROLE_ENV_PREFIX,
)
from lobes.profiles.shape_render import OPT_IN_ACTIVATION_ENV, OPT_IN_CORE_ACTIVATION_ENV

#: The committed lock's filename. Matches NEITHER the ``.env`` suffix rule nor
#: the ``.env.`` prefix rule of the repo's positional gitignore convention —
#: see the module docstring.
LOCK_FILENAME = "deployment.lock.toml"

#: Bumped when the on-disk schema changes shape. Readers (t7's ``--from-lock``,
#: t8's ``lock_drift``) should refuse a version they do not understand rather
#: than guess.
SCHEMA_VERSION = 1

# Suffixes profile_env appends to a role prefix that are NOT plain knob fields:
# `model` renders to two keys and an infeasible role renders the #110 marker.
_NON_KNOB_SUFFIXES = frozenset({"MODEL", "SERVED_NAME", "FEASIBLE"})

#: Keys the renderer DOES emit that the lock deliberately never records.
#: ``COMPOSE_PROFILES`` is operator-typed state (``_compose.MERGE_ONLY_FILES``
#: names it alongside the bearer key and the Hub token): a deployment's own
#: opt-in gears — ``muse``, ``worker``, ``minor`` — are declared there by the
#: operator, and a lock that re-stated a rendered value would fight them.
EXCLUDED_RENDERED_KEYS: frozenset[str] = frozenset({"COMPOSE_PROFILES"})

#: Key SUFFIXES the lock never records, whatever the renderer produces for
#: them. A lane's ``*_BASE_URL`` / ``*_URL`` is a WIRING fact, and while the
#: renderer only ever writes a compose-internal DNS name
#: (``http://vllm-worker:8000``), the same key is retargetable by hand at
#: another box — which would put an internal origin in a committed file, the
#: thing the spec's peer-origin exclusion exists to prevent. A restore
#: re-renders the wire from the shape anyway, so nothing is lost by leaving it
#: out.
EXCLUDED_KEY_SUFFIXES: tuple[str, ...] = ("_URL",)


def is_excluded(key: str) -> bool:
    """Whether *key* is deliberately kept out of the lock.

    This NARROWS the allowlist and can only ever remove keys — it is not a
    denylist in the dangerous sense, because nothing enters the lock by
    failing to match it. A key gets in only by being derived from the
    renderer's own tables in :func:`lock_keys`.
    """
    return key in EXCLUDED_RENDERED_KEYS or key.endswith(EXCLUDED_KEY_SUFFIXES)


_HEADER = (
    f"# {LOCK_FILENAME} — generated by lobes; do not hand-edit.\n"
    "# Secret-free BY CONSTRUCTION: the [env] table below is an allowlist of\n"
    "# keys lobes/profiles/render.py renders, never a copy of a deployed .env.\n"
    "# Re-capture it after `lobes switch` or a hand edit, or `lobes doctor`\n"
    "# will report lock_drift.\n"
)


@lru_cache(maxsize=1)
def lock_keys() -> frozenset[str]:
    """The allowlist: every ``.env`` key a committed lock may carry.

    Derived from three sources, none of them a hand-written list of names:

    1. the role-knob grammar — :data:`lobes.profiles.render.ROLE_ENV_PREFIX`
       crossed with the knob suffixes
       (:data:`lobes.profiles.render._KNOB_ENV_SUFFIX`) plus ``MODEL`` /
       ``SERVED_NAME`` / ``FEASIBLE``;
    2. every card-level ``host_env`` key declared by a packaged built-in
       profile (``LOBES_IOWAIT_DEGRADED_THRESHOLD`` on ``orin`` today);
    3. the activation keys an alternative-engine or opt-in lane renders
       (:data:`lobes.profiles.render.LLAMA_CPP_ACTIVATION_ENV`,
       :data:`lobes.profiles.shape_render.OPT_IN_ACTIVATION_ENV`,
       :data:`lobes.profiles.shape_render.OPT_IN_CORE_ACTIVATION_ENV`).

    :func:`is_excluded` then narrows the result — see
    :data:`EXCLUDED_RENDERED_KEYS` / :data:`EXCLUDED_KEY_SUFFIXES`.

    Adding a role, a knob or a ``host_env`` declaration widens this set
    automatically. Adding a *secret* does not: a credential is not a
    :class:`~lobes.profiles.schema.RoleProfile` field, so it can never appear
    here without someone deliberately declaring it as a rendered knob.
    """
    suffixes = set(_KNOB_ENV_SUFFIX.values()) | set(_NON_KNOB_SUFFIXES)
    keys = {f"{prefix}_{suffix}" for prefix in ROLE_ENV_PREFIX.values() for suffix in suffixes}
    for name in builtin_names():
        keys |= set(resolve_profile(name).host_env)
    for table in (
        LLAMA_CPP_ACTIVATION_ENV,
        OPT_IN_ACTIVATION_ENV,
        OPT_IN_CORE_ACTIVATION_ENV,
    ):
        for activation in table.values():
            keys |= set(activation)
    return frozenset(key for key in keys if not is_excluded(key))


def allowlist_env(env: Mapping[str, str]) -> dict[str, str]:
    """*env* filtered down to :func:`lock_keys`, sorted.

    Closed over its input by construction: a key the renderer does not produce
    changes nothing about the result, whatever it is called.
    """
    return {key: env[key] for key in sorted(env) if key in lock_keys()}


@dataclass(frozen=True)
class DeploymentLock:
    """One box's captured deployment state — the committed lock's in-memory form.

    :param variation: The variation id (machine type / setup, never a
        hostname). Resolved by the caller (t2's resolver), never detected here.
    :param env: The allowlisted rendered knobs — see :func:`allowlist_env`.
    :param profile: The card profile this variation resolved from, if known.
    :param shape: The deployment shape, if any.
    :param lobes_version: The ``lobes-cli`` version that captured the lock.
    :param files: Verbatim-committed deployment files, name -> content digest
        (see :func:`file_digest`) — what t8's ``lock_drift`` diffs against.
    :param evidence: A ``docs/evidence/`` transcript path, or ``None`` for "no
        measured result" (the #108 rule: never a blank a reader could mistake
        for a measurement).
    """

    variation: str
    env: Mapping[str, str]
    profile: str | None = None
    shape: str | None = None
    lobes_version: str | None = None
    files: Mapping[str, str] = None  # type: ignore[assignment]
    evidence: str | None = None

    def __post_init__(self) -> None:
        # Normalise the two mappings to plain sorted dicts so equality and
        # rendering are both order-independent. (Frozen dataclass: assign
        # through object.__setattr__.)
        object.__setattr__(self, "env", {key: self.env[key] for key in sorted(self.env)})
        files = self.files or {}
        object.__setattr__(self, "files", {key: files[key] for key in sorted(files)})


def build_lock(
    *,
    variation: str,
    env: Mapping[str, str],
    profile: str | None = None,
    shape: str | None = None,
    lobes_version: str | None = None,
    files: Mapping[str, str] | None = None,
    evidence: str | None = None,
) -> DeploymentLock:
    """Capture a deployment's state as a :class:`DeploymentLock`.

    *env* may be a whole deployed ``.env`` — secrets included — because it is
    passed through :func:`allowlist_env` here. That is the entire point: the
    caller never has to know which keys are sensitive.
    """
    return DeploymentLock(
        variation=variation,
        env=allowlist_env(env),
        profile=profile,
        shape=shape,
        lobes_version=lobes_version,
        files=dict(files or {}),
        evidence=evidence,
    )


def capture_lock(
    deploy_dir: os.PathLike | str,
    *,
    variation: str,
    profile: str | None = None,
    shape: str | None = None,
    lobes_version: str | None = None,
    files: Mapping[str, str] | None = None,
    evidence: str | None = None,
) -> DeploymentLock:
    """:func:`build_lock` over the ``.env`` of an existing deployment directory.

    Reading is the only filesystem action; nothing is written and the ``.env``
    is never modified (the merge-only guarantee
    :data:`lobes.runtime._compose.MERGE_ONLY_FILES` states).
    """
    from lobes.runtime._env import read_env_file

    return build_lock(
        variation=variation,
        env=read_env_file(Path(deploy_dir) / ".env"),
        profile=profile,
        shape=shape,
        lobes_version=lobes_version,
        files=files,
        evidence=evidence,
    )


def file_digest(path: os.PathLike | str) -> str:
    """``"sha256:<hex>"`` of a file's bytes — the ``[files]`` table's value form."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return f"sha256:{digest}"


# --- serialization -----------------------------------------------------------

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _toml_string(value: str) -> str:
    """*value* as a TOML basic string.

    Hand-rolled because :mod:`tomllib` is read-only and this repo's runtime is
    stdlib-only. The subset is small and fully covered: escape the five control
    shorthands plus backslash and quote, and ``\\uXXXX`` anything else below
    0x20 (or DEL). Real values exercise it — a lane's
    ``PRIMARY_SPECULATIVE_CONFIG`` carries both quotes and backslashes.
    """
    out = []
    for char in value:
        escape = _ESCAPES.get(char)
        if escape is not None:
            out.append(escape)
        elif char < "\x20" or char == "\x7f":
            out.append(f"\\u{ord(char):04X}")
        else:
            out.append(char)
    return '"' + "".join(out) + '"'


_BARE_KEY_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")


def _toml_key(key: str) -> str:
    """A bare key where TOML allows one, a quoted key otherwise (``docker-compose.yml``)."""
    if key and set(key) <= _BARE_KEY_CHARS:
        return key
    return _toml_string(key)


def _assert_secret_free(lock: DeploymentLock) -> None:
    """Refuse to render a lock carrying a key outside the allowlist.

    :func:`build_lock` cannot produce one, but a hand-constructed
    :class:`DeploymentLock` can — and this is the last gate before bytes reach
    a committed file, so it is checked here rather than trusted upstream.
    """
    stray = sorted(key for key in lock.env if key not in lock_keys())
    if stray:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=(
                f"refusing to write {LOCK_FILENAME}: {', '.join(stray)} "
                "is not a rendered knob key"
            ),
            remediation=(
                "build the lock with lobes.runtime._lock.build_lock(), which "
                "allowlists the deployed .env down to the keys "
                "lobes/profiles/render.py renders"
            ),
        )


def lock_toml(lock: DeploymentLock) -> str:
    """Render *lock* to the committed ``deployment.lock.toml`` text.

    Deterministic: both tables are emitted in sorted key order, so re-capturing
    an unchanged deployment produces a byte-identical file and any diff is a
    real change.
    """
    _assert_secret_free(lock)
    lines = [_HEADER, f"schema_version = {SCHEMA_VERSION}\n", "\n[variation]\n"]
    lines.append(f"id = {_toml_string(lock.variation)}\n")
    for field, value in (
        ("profile", lock.profile),
        ("shape", lock.shape),
        ("lobes_version", lock.lobes_version),
        ("evidence", lock.evidence),
    ):
        if value is not None:
            lines.append(f"{field} = {_toml_string(value)}\n")
    lines.append("\n[env]\n")
    for key in sorted(lock.env):
        lines.append(f"{_toml_key(key)} = {_toml_string(lock.env[key])}\n")
    if lock.files:
        lines.append("\n[files]\n")
        for name in sorted(lock.files):
            lines.append(f"{_toml_key(name)} = {_toml_string(lock.files[name])}\n")
    return "".join(lines)


def lock_path(directory: os.PathLike | str) -> Path:
    """Where the lock lives inside *directory*."""
    return Path(directory) / LOCK_FILENAME


def write_lock(directory: os.PathLike | str, lock: DeploymentLock) -> Path:
    """Write *lock* into *directory*; returns the path written.

    Mutation safety (dry-run by default, ``--apply`` to commit) is the CLI
    layer's convention and belongs to its callers (t7/t9) — this is the
    unconditional write they call once the operator has said yes.
    """
    text = lock_toml(lock)  # raises before any file is touched
    target = lock_path(directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def load_lock(path: os.PathLike | str) -> DeploymentLock:
    """Read a committed lock back into a :class:`DeploymentLock`.

    Refuses an unknown ``schema_version`` rather than guessing at fields it
    does not understand.
    """
    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    version = raw.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"{path}: unsupported lock schema_version {version!r}",
            remediation=f"this lobes understands schema_version {SCHEMA_VERSION}; upgrade lobes",
        )
    variation = raw.get("variation", {})
    return DeploymentLock(
        variation=variation.get("id", ""),
        env=dict(raw.get("env", {})),
        profile=variation.get("profile"),
        shape=variation.get("shape"),
        lobes_version=variation.get("lobes_version"),
        files=dict(raw.get("files", {})),
        evidence=variation.get("evidence"),
    )
