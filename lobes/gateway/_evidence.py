"""Host-measured evidence the gateway advertises for ``lobes site`` (t21, d2).

* **memory** -- ``/proc/meminfo`` MemTotal/MemAvailable, host-wide even inside
  docker on Linux; on ``GET /health`` as ``memory``, omitted when unreadable.

Deviation d2 (2026-11-10) removed the operator-typed split and tunnel
declarations t21 added (``<PREFIX>_SPANS``, ``<PREFIX>_WORLD_SIZE``,
``LOBES_TUNNEL_PUBLIC_URL``): hand-typed facts drift from the running system.
``lobes site`` now reads split brains and the outside-tunnel hub from each
member's running containers (``lobes/site/_topology.py``). With memory
unreadable the body is byte-identical to the original three keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

MEMINFO = Path("/proc/meminfo")


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


def health_payload(
    env: Mapping[str, str] | None = None, *, version: str, meminfo: Path | None = None
) -> dict:
    """The ``GET /health`` body: the original three keys plus measured ``memory``.

    ``env`` is accepted for call-site stability and not read: nothing on
    ``/health`` is operator-declared any more (deviation d2).
    """
    body: dict[str, object] = {
        "status": "ok",
        "service": "model-gear-gateway",
        "version": version,
    }
    memory = read_meminfo(meminfo)
    if memory is not None:
        body["memory"] = memory
    return body
