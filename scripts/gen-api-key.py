#!/usr/bin/env python3
"""Generate (or rotate) the bearer key that gates the served vLLM API.

Writes ``CULTURE_VLLM_API_KEY`` into the deployment dir's ``.env`` (the value vLLM
reads as ``VLLM_API_KEY``, requiring ``Authorization: Bearer <key>`` on every
request). The secret is generated fresh with the stdlib :mod:`secrets` module and
is **never** hardcoded here — so this script is safe to keep in the open-source
repo, and the key only ever lives in the gitignored deployment ``.env``.

By default the key is written but **not printed** (so it doesn't leak into logs
or terminal scrollback); read it back privately from ``.env`` to configure
clients, or pass ``--show`` to emit it on stdout.

Deployment dir resolves like the ``model`` CLI: ``--dir`` → ``$MODEL_GEAR_DIR`` →
``~/.model-gear``. Stdlib only; no model_gear import, so it runs from a wheel
install too.

Examples:
    python3 scripts/gen-api-key.py                 # set if absent (no echo)
    python3 scripts/gen-api-key.py --force         # rotate existing (no echo)
    python3 scripts/gen-api-key.py --force --show   # rotate and print the new key
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

KEY = "CULTURE_VLLM_API_KEY"
PREFIX = "mg-"  # human-readable provenance marker; not a secret


def _deploy_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env = os.environ.get("MODEL_GEAR_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".model-gear"


def _read_key(env_path: Path) -> str | None:
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    prefix = KEY + "="
    for line in text.splitlines():
        if line.startswith(prefix):
            value = line[len(prefix) :]
            return value or None
    return None


def _write_key(env_path: Path, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    out: list[str] = []
    seen = False
    for line in lines:
        if line.startswith(KEY + "="):
            out.append(f"{KEY}={value}")
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append(f"{KEY}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.chmod(env_path, 0o600)  # the .env holds a secret — keep it owner-only


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"Generate/rotate {KEY} in the deployment .env (open-source-safe).",
    )
    parser.add_argument("--dir", help="Deployment dir (default: $MODEL_GEAR_DIR or ~/.model-gear).")
    parser.add_argument("--force", action="store_true", help="Rotate even if a key already exists.")
    parser.add_argument(
        "--bytes", type=int, default=32, help="Token entropy in bytes (default: 32)."
    )
    parser.add_argument(
        "--show", action="store_true", help="Print the key on stdout (else hidden)."
    )
    args = parser.parse_args(argv)

    env_path = _deploy_dir(args.dir) / ".env"
    if not env_path.parent.is_dir():
        print(
            f"error: deployment dir {env_path.parent} not found",
            file=sys.stderr,
        )
        print("hint: run 'model init --apply' to scaffold it first", file=sys.stderr)
        return 2

    existing = _read_key(env_path)
    if existing and not args.force:
        print(f"error: {KEY} is already set in {env_path}", file=sys.stderr)
        print("hint: pass --force to rotate it", file=sys.stderr)
        return 1

    token = PREFIX + secrets.token_urlsafe(args.bytes)
    _write_key(env_path, token)

    verb = "rotated" if existing else "set"
    print(f">> {verb} {KEY} in {env_path}", file=sys.stderr)
    print(">> restart to enforce it: model serve --apply", file=sys.stderr)
    if args.show:
        print(token)  # stdout = the key, for capture/piping when explicitly requested
    else:
        print(">> key hidden; read it privately from .env to configure clients", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
