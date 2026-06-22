"""Read and write the ``.env`` file in a deployment directory (stdlib only).

Ported from the ``_get_env`` / ``_set_env`` helpers in the original
``model-runner.sh``. Same semantics: an empty value (``KEY=``) reads as the
caller's default, mirroring bash's ``${v:-default}``.
"""

from __future__ import annotations

import os
from pathlib import Path

from lobes.cli._errors import EXIT_ENV_ERROR, ModelGearError


def read_env(env_path: os.PathLike | str, key: str, default: str | None = None) -> str | None:
    """Return the value of ``key`` in the ``.env`` file.

    Falls back to ``default`` when the file is unreadable, the key is absent, or
    the value is empty (``KEY=``) — matching the shell's ``${v:-default}``.
    """
    try:
        text = Path(env_path).read_text(encoding="utf-8")
    except OSError:
        return default
    prefix = key + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :]
            return value if value else default
    return default


def parse_port(value: object, source: str = "VLLM_PORT") -> int:
    """Parse a port to ``int``, turning a bad value into a structured error.

    Without this a non-numeric ``VLLM_PORT`` in ``.env`` (or a stray ``--port``)
    surfaces as the dispatcher's generic ``unexpected: ValueError``.
    """
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"invalid port {value!r} from {source}",
            remediation="set a numeric VLLM_PORT in .env, or pass --port N",
        ) from exc


def set_env(env_path: os.PathLike | str, key: str, value: str) -> None:
    """Update ``KEY=VALUE`` in ``.env`` (rewrite if present, append if absent)."""
    path = Path(env_path)
    if not path.is_file():
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f".env not found at {path}",
            remediation="run 'lobes init --apply' first",
        )
    prefix = key + "="
    out: list[str] = []
    seen = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(prefix):
            out.append(f"{key}={value}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
