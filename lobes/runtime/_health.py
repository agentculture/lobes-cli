"""Poll the vLLM ``/health`` endpoint (stdlib ``urllib``).

Ported from ``_wait_health`` in the original ``model-runner.sh``: poll every
15s, fail fast if the container exits, and surface the last logs on failure.
"""

from __future__ import annotations

import json
import time
import urllib.request

from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError
from lobes.cli._output import emit_diagnostic
from lobes.runtime import _compose


def is_healthy(port: int, timeout: float = 3.0) -> bool:
    """Non-blocking: True if ``/health`` returns a 2xx within ``timeout``."""
    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # local endpoint only
            return 200 <= r.status < 300
    except OSError:  # URLError subclasses OSError
        return False


def fetch_health(port: int, timeout: float = 3.0) -> dict | None:
    """Non-blocking: the parsed JSON body of ``/health``, or ``None`` on any failure.

    :func:`is_healthy` collapses ``/health`` to a bare up/down boolean; this
    returns the payload itself so a caller can inspect what the gateway
    *reports* beyond bare liveness — e.g. its own deployed ``lobes-cli``
    ``version`` (issue #99), which ``lobes doctor`` compares against the CLI's
    own :data:`lobes.__version__` to catch deployed-artifact skew (the
    structural cause of issue #92: a gateway container built from a stale pin
    silently outliving a PyPI release that already fixed the bug it was
    misdiagnosed as).

    ``None`` covers every failure mode uniformly — connection refused, DNS
    failure, timeout, a non-2xx status, and a malformed/non-JSON or non-object
    body — because a caller here only ever needs one distinction: "the gateway
    told us something" vs "we could not learn anything", never *why* it
    couldn't be reached.
    """
    url = f"http://localhost:{port}/health"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # local endpoint only
            if not (200 <= r.status < 300):
                return None
            raw = r.read()
    except OSError:  # URLError subclasses OSError
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def wait_health(
    port: int,
    *,
    deadline_seconds: int = 2700,
    interval: int = 15,
    container: str = _compose.CONTAINER,
) -> None:
    """Block until ``/health`` responds, or raise on timeout / container exit."""
    deadline = time.monotonic() + deadline_seconds
    emit_diagnostic(
        f">> waiting for /health on :{port} (first run downloads weights; up to "
        f"{deadline_seconds // 60} min)"
    )
    while True:
        if is_healthy(port):
            emit_diagnostic(f">> healthy on :{port}")
            return
        if time.monotonic() >= deadline:  # pragma: no cover - real-time deadline
            raise ModelGearError(
                code=EXIT_ENV_ERROR,
                message=f"timeout waiting for health on :{port}",
                remediation=f"check 'lobes status' and 'docker logs {container}'",
            )
        state = _compose.container_status(container)
        if state != "running":
            logs = _compose.container_logs(container, tail=30)
            raise ModelGearError(
                code=EXIT_ENV_ERROR,
                message=f"container is '{state}' — load failed",
                remediation=(
                    "recent logs:\n" + logs if logs else "see 'docker logs " + container + "'"
                ),
            )
        time.sleep(interval)  # pragma: no cover - real-time poll
