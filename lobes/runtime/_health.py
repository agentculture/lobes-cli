"""Poll the vLLM ``/health`` endpoint (stdlib ``urllib``).

Ported from ``_wait_health`` in the original ``model-runner.sh``: poll every
15s, fail fast if the container exits, and surface the last logs on failure.
"""

from __future__ import annotations

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
