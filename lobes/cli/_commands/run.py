"""``lobes run minor`` — send a prompt to the minor lobe and print the reply.

Read-only: never writes ``.env``, docker-compose, or any repo file.
No ``--apply`` flag needed or accepted.

Usage::

    lobes run minor "<prompt>"
    lobes run minor "<prompt>" --system "You are a helpful assistant."
    lobes run minor "<prompt>" --max-tokens 256
    lobes run minor "<prompt>" --json

The minor lobe model is resolved from the supported-model catalog
(``role_hint == "minor"``).  Override with ``--model <id>`` when the
catalog entry does not yet exist or you want to target a specific model id.

The gateway base URL defaults to ``http://localhost:8000/v1`` (the local
fleet endpoint).  Override with ``--base-url`` to target a remote deployment.
"""

from __future__ import annotations

import argparse

from lobes.catalog import supported_models
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.minor import chat_completion, chat_text

_DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _resolve_model(args: argparse.Namespace) -> str:
    """Resolve the model id: ``--model`` wins, else catalog lookup by role_hint.

    Raises :class:`~lobes.cli._errors.ModelGearError` when no ``--model`` was
    given and the catalog has no entry with ``role_hint == "minor"``.
    """
    explicit = getattr(args, "model", None)
    if explicit:
        return explicit
    models = [m for m in supported_models() if m.role_hint == "minor"]
    if not models:
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message="no model with role_hint='minor' found in the catalog",
            remediation="pass --model <model-id> to target a specific model id",
        )
    return models[0].id


def cmd_run_minor(args: argparse.Namespace) -> int:
    """Handler for ``lobes run minor``."""
    json_mode = bool(getattr(args, "json", False))

    # Validate the lobe name — only "minor" is currently supported.
    lobe: str = args.lobe
    if lobe != "minor":
        raise ModelGearError(
            code=EXIT_USER_ERROR,
            message=f"unsupported lobe: {lobe!r} (only 'minor' is currently supported)",
            remediation='usage: lobes run minor "<prompt>"',
        )

    model_id = _resolve_model(args)
    base_url: str = getattr(args, "base_url", None) or _DEFAULT_BASE_URL
    system: str | None = getattr(args, "system", None)
    max_tokens: int | None = getattr(args, "max_tokens", None)

    if json_mode:
        result = chat_completion(
            args.prompt,
            base_url=base_url,
            model=model_id,
            system=system,
            max_tokens=max_tokens,
        )
        emit_result(result, json_mode=True)
    else:
        text = chat_text(
            args.prompt,
            base_url=base_url,
            model=model_id,
            system=system,
            max_tokens=max_tokens,
        )
        emit_result(text, json_mode=False)

    return 0


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``run`` verb into *sub* (the top-level subparsers action).

    This module intentionally does NOT import or modify ``lobes.cli.__init__``;
    wiring into the main parser is a separate concern (task t8).
    """
    p = sub.add_parser(
        "run",
        help=(
            "Read-only: send a prompt to a lobe and print the reply "
            "(e.g. 'lobes run minor \"<prompt>\"')."
        ),
    )
    p.add_argument(
        "lobe",
        help="Lobe to target.  Currently only 'minor' is supported.",
    )
    p.add_argument(
        "prompt",
        help="The user prompt to send to the lobe.",
    )
    p.add_argument(
        "--base-url",
        dest="base_url",
        default=_DEFAULT_BASE_URL,
        help=(
            f"OpenAI-compatible base URL of the local fleet gateway "
            f"(default: {_DEFAULT_BASE_URL})."
        ),
    )
    p.add_argument(
        "--model",
        default=None,
        help="Override the model id (default: resolved from the catalog by role_hint='minor').",
    )
    p.add_argument(
        "--system",
        default=None,
        help="Optional system message prepended before the user prompt.",
    )
    p.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help=(
            "Emit the full chat-completion JSON object to stdout "
            "instead of just the assistant text."
        ),
    )
    p.set_defaults(func=cmd_run_minor)
