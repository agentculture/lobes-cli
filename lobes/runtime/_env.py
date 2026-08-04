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


def read_env_file(env_path: os.PathLike | str) -> dict[str, str]:
    """Parse a whole ``.env`` file into a plain ``dict`` (``{}`` if unreadable).

    Same ``KEY=VALUE`` line format as :func:`read_env`/:func:`set_env` — no
    quoting, no variable expansion; blank lines and ``#`` comments are skipped.
    Lets a host-side, read-only caller (``lobes capabilities`` / ``lobes
    endpoint``, issue #81 t5) build an env ``Mapping`` for
    :func:`lobes.roles.role_registry_from_env` from a deployment's ``.env``
    without hand-listing every key the registry might read — the CLI process
    never has the deployment's env vars in its own ``os.environ`` (those are
    injected into the *containers* by ``docker compose``, not the host shell).
    """
    try:
        text = Path(env_path).read_text(encoding="utf-8")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.split("#", 1)[0].strip()
        if not stripped or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key:
            out[key] = value.strip()
    return out


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
    # Last-resort guardrail: this file is line-oriented, so a value carrying a
    # newline would split one entry into two physical lines and corrupt every
    # subsequent read. Callers should validate earlier (profile `host_env` does),
    # but refusing here means no code path can write a malformed .env at all.
    if any(ch in value for ch in ("\n", "\r", "\x00")):
        raise ModelGearError(
            code=EXIT_ENV_ERROR,
            message=f"refusing to write {key}: the value contains a newline or NUL",
            remediation="`.env` is line-oriented KEY=VALUE — keep values on one line",
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
