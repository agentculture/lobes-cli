"""``lobes route "<text>"`` — classify a task into a catalog gear routing decision.

Read-only: never writes ``.env``, docker-compose, or any repo file.
No ``--apply`` flag needed or accepted.

Usage::

    lobes route "<task description>"
    lobes route "<task description>" --json
    lobes route "<task description>" --base-url http://other:8000/v1
    lobes route "<task description>" --model <model-id>

The minor lobe model (``role_hint == "minor"``) is asked to classify the task
into one of the catalog gear roles. Governance is overlaid via
:func:`lobes.minor.decide` so any escalation condition forces ``escalate=True``
regardless of the model's suggestion. Routing targets are **only** lobes catalog
gear roles — not tools, not mesh agents.

Output schema::

    {
        "chosen_gear": str,   # catalog gear role (e.g. "minor"/"primary"/"candidate"/...)
        "escalate": bool,     # True when governance forces escalation
        "confidence": float,  # [0,1] — model's self-reported confidence, clamped
        "reason": str,        # human-readable explanation
    }

Governance detail
-----------------
:func:`lobes.minor.decide` is called with ``duty="route"`` (an allowed duty)
and ``conditions`` extracted from the model's response. Any recognised
:data:`lobes.minor.ESCALATION_CONDITIONS` string forces ``escalate=True``; the
model's gear suggestion is preserved but agents MUST honour the escalation flag.
"""

from __future__ import annotations

import argparse
import json
import re

from lobes.catalog import supported_models
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError
from lobes.cli._output import emit_result
from lobes.minor import chat_completion, decide

_DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Sane confidence default when the model does not self-report one.
_DEFAULT_CONFIDENCE: float = 0.7

# A routing decision is a short JSON object — cap generation so a thinking-mode
# model can't run past the client timeout emitting a long <think> trace, and
# disable thinking outright so the reply is the JSON we asked for (verified live:
# enable_thinking=false returns a terse parseable object in well under a second).
_ROUTE_MAX_TOKENS: int = 512
_ROUTE_EXTRA_BODY: dict = {"chat_template_kwargs": {"enable_thinking": False}}

# ---------------------------------------------------------------------------
# System prompt for the routing classifier
# ---------------------------------------------------------------------------

_ROUTE_SYSTEM: str = (
    "You are a routing classifier for the lobes model fleet. "
    "Given a task description, classify it into the most appropriate catalog gear role.\n\n"
    "Available catalog gear roles:\n"
    '- "minor": Small 4B model. Best for: quick formatting, validation, '
    "classification, suggestion, summarization.\n"
    '- "primary": Default 27B primary (text-only, MTP speculative decoding). '
    "Best for: complex reasoning, generation, code, most tasks.\n"
    '- "candidate": Alternative 27B/32B models. Best for: vision tasks or '
    "when the primary is unavailable.\n"
    '- "fallback": 24B Mistral model. Best for: when primary/candidate are '
    "unavailable; vision-capable.\n"
    '- "embedding": 0.6B embedding model. Best for: text embeddings, '
    "semantic search, similarity.\n"
    '- "reranker": 0.6B cross-encoder. Best for: re-ranking search results, '
    "passage scoring.\n\n"
    "Also identify any applicable escalation conditions from this set "
    "(include only the ones that clearly apply):\n"
    '- "needs_codebase_context"\n'
    '- "security_sensitive"\n'
    '- "architectural_decision"\n'
    '- "write_or_delete_operation"\n'
    '- "final_review_required"\n\n'
    "Respond with ONLY a valid JSON object — no markdown, no explanation, "
    "no surrounding text:\n"
    "{\n"
    '  "chosen_gear": "<role>",\n'
    '  "confidence": <float between 0.0 and 1.0>,\n'
    '  "reason": "<one-sentence explanation>",\n'
    '  "conditions": [<zero or more applicable escalation condition strings>]\n'
    "}"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_model(args: argparse.Namespace) -> str:
    """Resolve the minor model id: ``--model`` wins, else catalog lookup by role_hint.

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


def _parse_model_response(content: str) -> dict:
    """Parse the model's routing response JSON.

    Strategy (fail-safe):
    1. Try to parse the entire content as JSON.
    2. Try to extract the first ``{...}`` block via regex.
    3. Fall back to a safe default (primary, low confidence).
    """
    # 1. Direct JSON parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # 2. Regex extraction of the first { … } block
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # 3. Safe fallback
    return {
        "chosen_gear": "primary",
        "confidence": _DEFAULT_CONFIDENCE,
        "reason": "Could not parse model response; defaulting to primary.",
        "conditions": [],
    }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------


def cmd_route(args: argparse.Namespace) -> int:
    """Handler for ``lobes route``."""
    json_mode = bool(getattr(args, "json", False))
    model_id = _resolve_model(args)
    base_url: str = getattr(args, "base_url", None) or _DEFAULT_BASE_URL

    # Ask the minor model to classify the task.
    completion = chat_completion(
        args.text,
        base_url=base_url,
        model=model_id,
        system=_ROUTE_SYSTEM,
        max_tokens=_ROUTE_MAX_TOKENS,
        extra_body=_ROUTE_EXTRA_BODY,
    )
    content: str = completion["choices"][0]["message"]["content"]
    parsed = _parse_model_response(content)

    # Extract fields from the model's response.
    chosen_gear: str = str(parsed.get("chosen_gear") or "primary")
    raw_confidence = parsed.get("confidence")
    model_reason: str = str(parsed.get("reason") or "No reason provided.")
    conditions: list[str] = list(parsed.get("conditions") or [])

    # Clamp confidence to [0, 1].  Use the sane default when the model
    # omits or nulls the field.
    try:
        raw_float = float(raw_confidence) if raw_confidence is not None else _DEFAULT_CONFIDENCE
    except (TypeError, ValueError):
        raw_float = _DEFAULT_CONFIDENCE
    confidence: float = _clamp(raw_float)

    # Overlay governance — duty="route" is in ALLOWED, but any recognised
    # escalation condition still forces escalate=True.
    gov = decide(duty="route", conditions=conditions)

    # Build the combined reason string.
    if gov.escalate:
        reason = f"{gov.reason} (model suggested: {model_reason})"
    else:
        reason = model_reason

    decision: dict = {
        "chosen_gear": chosen_gear,
        "escalate": gov.escalate,
        "confidence": confidence,
        "reason": reason,
    }

    if json_mode:
        emit_result(decision, json_mode=True)
    else:
        escalate_str = "yes" if gov.escalate else "no"
        summary = (
            f"chosen_gear: {chosen_gear}\n"
            f"escalate:    {escalate_str}\n"
            f"confidence:  {confidence:.2f}\n"
            f"reason:      {reason}\n"
        )
        emit_result(summary, json_mode=False)

    return 0


# ---------------------------------------------------------------------------
# Verb registration
# ---------------------------------------------------------------------------


def register(sub: argparse._SubParsersAction) -> None:
    """Register the ``route`` verb into *sub* (the top-level subparsers action).

    This module intentionally does NOT import or modify ``lobes.cli.__init__``;
    wiring into the main parser is a separate concern (task t8).
    """
    p = sub.add_parser(
        "route",
        help=(
            "Read-only: classify a task description into a catalog gear routing decision "
            "(e.g. 'lobes route \"summarize this PR\"')."
        ),
    )
    p.add_argument(
        "text",
        help="Task description to route to the appropriate catalog gear.",
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
        help=(
            "Override the model id used for classification "
            "(default: resolved from the catalog by role_hint='minor')."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the routing decision as a JSON object to stdout.",
    )
    p.set_defaults(func=cmd_route)
