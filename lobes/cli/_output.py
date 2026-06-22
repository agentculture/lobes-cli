"""stdout / stderr helpers with a strict split.

Rule: **results go to stdout, diagnostics and errors go to stderr.** Agents
parsing lobes output can rely on this invariant. JSON mode routes structured
payloads to the same streams — never mixes them.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from lobes.cli._errors import ModelGearError


def emit_result(data: Any, *, json_mode: bool, stream: TextIO | None = None) -> None:
    """Write a command result.

    Text mode: ``data`` is treated as a string (or stringified) and a trailing
    newline is ensured. JSON mode: ``data`` is JSON-dumped with a trailing
    newline. Default stream is stdout.
    """
    s = stream if stream is not None else sys.stdout
    if json_mode:
        json.dump(data, s, ensure_ascii=False)
        s.write("\n")
        return
    text = data if isinstance(data, str) else str(data)
    s.write(text)
    if not text.endswith("\n"):
        s.write("\n")


def emit_error(err: ModelGearError, *, json_mode: bool, stream: TextIO | None = None) -> None:
    """Write a :class:`ModelGearError` to stderr.

    Text mode renders as two lines when a remediation is present::

        error: <message>
        hint: <remediation>
    """
    s = stream if stream is not None else sys.stderr
    if json_mode:
        json.dump(err.to_dict(), s, ensure_ascii=False)
        s.write("\n")
        return
    s.write(f"error: {err.message}\n")
    if err.remediation:
        s.write(f"hint: {err.remediation}\n")


def emit_diagnostic(message: str, *, stream: TextIO | None = None) -> None:
    """Write a human diagnostic (progress, summary) to stderr."""
    s = stream if stream is not None else sys.stderr
    s.write(message if message.endswith("\n") else message + "\n")
