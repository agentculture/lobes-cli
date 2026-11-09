"""Release preflight: is ``lobes-cli==<v>`` resolvable from the gateway base image?

Publishing to PyPI is followed by CDN lag (observed 2026-11-09), so a gateway
rebuild started right after a release can 404. This check runs
``docker run --rm <base> pip download --no-deps lobes-cli==<v>`` with
exponential backoff and a bounded total wait, BEFORE any member is touched.

By design this module takes and calls no member transport: it can only talk to
docker (through an injected runner), so a failed preflight contacts no member.
"""

from __future__ import annotations

import os
import re
import subprocess  # nosec
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

#: Runner contract: argv -> (returncode, combined output).
Runner = Callable[[Sequence[str]], "tuple[int, str]"]

DEFAULT_TOTAL_S = 300.0
DEFAULT_INITIAL_S = 5.0
DEFAULT_FACTOR = 2.0
DEFAULT_MAX_DELAY_S = 60.0
DEV_INDEX_ENV = "LOBES_DEV_INDEX_URL"

_DOCKERFILE = Path(__file__).resolve().parent.parent / "templates" / "fleet" / "Dockerfile.gateway"
_FROM_RE = re.compile(r"^\s*FROM\s+(?:--\S+\s+)*(\S+)", re.IGNORECASE | re.MULTILINE)


class PreflightError(RuntimeError):
    """The gateway Dockerfile could not be understood."""


@dataclass(frozen=True)
class PreflightResult:
    ok: bool
    version: str
    base_image: str
    attempts: int
    waited_s: float
    hint: str = ""
    last_output: str = field(default="", repr=False)


def gateway_base_image(dockerfile: Optional[Path] = None) -> str:
    """Read the base image from Dockerfile.gateway's first FROM line."""
    path = dockerfile or _DOCKERFILE
    match = _FROM_RE.search(path.read_text(encoding="utf-8"))
    if not match:
        raise PreflightError(f"no FROM line in {path}")
    return match.group(1)


def dev_index_url(environ: Optional[dict] = None) -> str:
    """LOBES_DEV_INDEX_URL, as the Dockerfile's build arg (empty = unset)."""
    env = os.environ if environ is None else environ
    return env.get(DEV_INDEX_ENV, "") or ""


def build_command(base: str, version: str, index_url: str = "") -> list:
    """argv for the resolvability probe; mirrors the Dockerfile's dev-lane step."""
    cmd = ["docker", "run", "--rm", base, "pip", "download", "--no-deps"]
    cmd += ["--quiet"]
    if index_url:
        cmd += ["--index-url", index_url]
    cmd.append(f"lobes-cli=={version}")
    return cmd


def _default_runner(argv: Sequence[str]) -> "tuple[int, str]":
    proc = subprocess.run(  # nosec
        list(argv), capture_output=True, text=True, check=False, timeout=600
    )
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def check_release(
    version: str,
    *,
    runner: Optional[Runner] = None,
    sleep: Callable[[float], None] = time.sleep,
    dockerfile: Optional[Path] = None,
    environ: Optional[dict] = None,
    total_s: float = DEFAULT_TOTAL_S,
    initial_s: float = DEFAULT_INITIAL_S,
    factor: float = DEFAULT_FACTOR,
    max_delay_s: float = DEFAULT_MAX_DELAY_S,
) -> PreflightResult:
    """Retry the probe with exponential backoff until it passes or ``total_s`` of
    (virtual, sleep-accounted) waiting is spent. Never contacts a member."""
    run = runner or _default_runner
    base = gateway_base_image(dockerfile)
    index = dev_index_url(environ)
    cmd = build_command(base, version, index)
    waited = 0.0
    delay = initial_s
    attempts = 0
    output = ""
    while True:
        attempts += 1
        code, output = run(cmd)
        if code == 0:
            return PreflightResult(True, version, base, attempts, waited, last_output=output)
        remaining = total_s - waited
        if remaining <= 0:
            break
        pause = min(delay, max_delay_s, remaining)
        sleep(pause)
        waited += pause
        delay *= factor
    where = f"index {index}" if index else "PyPI"
    hint = (
        f"lobes-cli=={version} did not resolve from {where} inside {base} after "
        f"{attempts} attempts over {waited:.0f}s. Check the release is published "
        f"(CDN lag can take minutes) or set {DEV_INDEX_ENV} for a .devN build. "
        "No member was contacted."
    )
    return PreflightResult(False, version, base, attempts, waited, hint, output)
