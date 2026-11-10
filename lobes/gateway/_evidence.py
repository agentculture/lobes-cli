"""Operator-declared and host-measured evidence the gateway advertises (t21).

Three facts a reader of the mesh (``lobes site``) cannot learn otherwise:

* **split** -- a generate role served by ONE engine spanning several boxes
  (e.g. TensorFold across two DGX Sparks). Declared per role as
  ``<PREFIX>_SPANS`` (comma-separated mesh member names, at least two) and/or
  ``<PREFIX>_WORLD_SIZE`` (int >= 1), and published as the role's optional
  ``spans`` / ``world_size`` keys on ``GET /capabilities``. Operator-typed,
  never derived (#92). They are NOT part of the mesh fingerprint: pooling
  and verification read only ``fingerprint`` (see
  :func:`lobes.gateway._mesh_routing.fingerprints_identical`).
* **memory** -- ``/proc/meminfo`` MemTotal/MemAvailable, host-wide even inside
  docker on Linux; on ``GET /health`` as ``memory``, omitted when unreadable.
* **tunnel** -- ``LOBES_TUNNEL_PUBLIC_URL``, the outside URL this box's tunnel
  publishes; on ``GET /health`` as ``tunnel: {public_url, declared: true}``
  only when set. Declared, not probed.

A malformed declaration is ignored with one stderr line (once per key/value),
never a crash. With none of the keys set every response is byte-identical to
before.
"""

from __future__ import annotations

import re
import sys
import threading
from collections.abc import Mapping
from pathlib import Path

MEMINFO = Path("/proc/meminfo")
TUNNEL_ENV = "LOBES_TUNNEL_PUBLIC_URL"

# The generate roles with a per-backend env prefix (``<PREFIX>_<KNOB>``, the
# same prefix as ``PRIMARY_MAX_MODEL_LEN`` etc.: the backend name upper-cased).
SPLIT_ROLE_PREFIX: dict[str, str] = {
    "cortex": "PRIMARY",
    "senses": "MULTIMODAL",
    "muse": "MUSE",
    "worker": "WORKER",
    "associate": "ASSOCIATE",
    "hand": "HAND",
}

# A mesh member name: it lands in served model names ({role}-{member}), so
# the same conservative shape the site accepts.
MEMBER_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_WARNED: set[tuple[str, str]] = set()
_WARN_LOCK = threading.Lock()


def _warn(key: str, value: str, why: str) -> None:
    with _WARN_LOCK:
        if (key, value) in _WARNED:
            return
        _WARNED.add((key, value))
    sys.stderr.write(f"lobes gateway: {key}={value!r} ignored: {why}\n")


def _spans(key: str, raw: str) -> list[str] | None:
    names = [n.strip() for n in raw.split(",")]
    if any(not n for n in names):
        _warn(key, raw, "empty member name")
        return None
    bad = [n for n in names if not MEMBER_NAME_RE.fullmatch(n)]
    if bad:
        _warn(key, raw, f"not a mesh member name: {bad[0]!r}")
        return None
    unique = list(dict.fromkeys(names))
    if len(unique) < 2:
        _warn(key, raw, "a split needs at least two distinct member names")
        return None
    return unique


def _world_size(key: str, raw: str) -> int | None:
    try:
        n = int(raw)
    except ValueError:
        _warn(key, raw, "not an integer")
        return None
    if n < 1:
        _warn(key, raw, "must be >= 1")
        return None
    return n


def split_facts(env: Mapping[str, str]) -> dict[str, dict[str, object]]:
    """``role -> {"spans": [...], "world_size": N}``, declared keys only."""
    out: dict[str, dict[str, object]] = {}
    for role, prefix in SPLIT_ROLE_PREFIX.items():
        facts: dict[str, object] = {}
        spans_key, ws_key = f"{prefix}_SPANS", f"{prefix}_WORLD_SIZE"
        raw = (env.get(spans_key) or "").strip()
        if raw:
            spans = _spans(spans_key, raw)
            if spans is not None:
                facts["spans"] = spans
        raw = (env.get(ws_key) or "").strip()
        if raw:
            ws = _world_size(ws_key, raw)
            if ws is not None:
                facts["world_size"] = ws
        if facts:
            out[role] = facts
    return out


def annotate_split(payload: dict, env: Mapping[str, str]) -> dict:
    """Add declared ``spans``/``world_size`` to each role entry in place."""
    for role, facts in split_facts(env).items():
        entry = payload.get(role)
        if isinstance(entry, dict):
            entry.update(facts)
    return payload


def read_meminfo(path: Path | None = None) -> dict[str, int] | None:
    """``{"total_bytes", "available_bytes"}`` from meminfo, or None."""
    try:
        text = (path or MEMINFO).read_text(encoding="utf-8")
    except OSError:
        return None
    kb: dict[str, int] = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        bits = rest.split()
        if bits and bits[0].isdigit():
            kb[key.strip()] = int(bits[0]) * 1024
    if "MemTotal" not in kb or "MemAvailable" not in kb:
        return None
    return {"total_bytes": kb["MemTotal"], "available_bytes": kb["MemAvailable"]}


def health_payload(env: Mapping[str, str], *, version: str, meminfo: Path | None = None) -> dict:
    """The ``GET /health`` body: the original three keys plus the evidence."""
    body: dict[str, object] = {
        "status": "ok",
        "service": "model-gear-gateway",
        "version": version,
    }
    memory = read_meminfo(meminfo)
    if memory is not None:
        body["memory"] = memory
    url = (env.get(TUNNEL_ENV) or "").strip().rstrip("/")
    if url:
        body["tunnel"] = {"public_url": url, "declared": True}
    return body
