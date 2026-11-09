"""Secret redaction for rollout output (stdlib only)."""

from __future__ import annotations

import re
from typing import Iterable

REDACTED = "[REDACTED]"

# NAME=value / NAME: value / "NAME": "value" for secret-looking names.
_NAME = r"[A-Za-z0-9_]*(?:API_KEY|_TOKEN|MESH_KEY|GATEWAY_KEY)|GATEWAY_API_KEY|LOBES_MESH_KEY|[A-Za-z0-9_]*TOKEN"
_ASSIGN = re.compile(
    r"(?P<name>\b(?:" + _NAME + r")\b[\"']?\s*[=:]\s*)(?P<q>[\"']?)(?P<val>[^\s\"',;]+)(?P=q)"
)
_BEARER = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+")


def redact(text: str, secrets: Iterable[str] = ()) -> str:
    """Remove secret values from text: known values plus KEY=value patterns."""
    if not text:
        return text
    out = text
    for s in sorted({s for s in secrets if s}, key=len, reverse=True):
        out = out.replace(s, REDACTED)
    out = _ASSIGN.sub(lambda m: f"{m.group('name')}{m.group('q')}{REDACTED}{m.group('q')}", out)
    return _BEARER.sub(lambda m: m.group(1) + REDACTED, out)
