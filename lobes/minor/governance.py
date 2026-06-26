"""Governance and escalation model for the **minor** role.

This module encodes *what* the minor lobe MAY do vs MUST escalate, keyed
entirely to the *role name* ``"minor"`` — never to a specific model identifier.
Swapping the underlying model is a catalog-only change; nothing here needs to
change.

Public API
----------
ROLE : str
    The role name this governance policy applies to (``"minor"``).
ALLOWED : frozenset[str]
    Duties the minor lobe may perform locally without escalation.
FORBIDDEN : frozenset[str]
    Actions the minor lobe must never perform; always trigger escalation.
ESCALATION_CONDITIONS : frozenset[str]
    Runtime conditions that force escalation regardless of the duty.
Decision
    Frozen dataclass returned by :func:`decide`.
decide(*, duty=None, conditions=()) -> Decision
    Evaluate a proposed action against this governance policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# ---------------------------------------------------------------------------
# Role identity
# ---------------------------------------------------------------------------

ROLE: str = "minor"
"""The role name this governance policy applies to.

Governance is role-keyed, not model-keyed, so that the underlying model can be
swapped in the catalog without touching this file.
"""

# ---------------------------------------------------------------------------
# Duty / action catalogs
# ---------------------------------------------------------------------------

ALLOWED: frozenset[str] = frozenset(
    {
        "prepare",
        "classify",
        "format",
        "validate",
        "suggest",
        "summarize",
        "route",
    }
)
"""Duties the minor lobe may perform locally (no escalation required)."""

FORBIDDEN: frozenset[str] = frozenset(
    {
        "approve",
        "finalize",
        "delete",
        "deploy",
        "architectural_decision",
    }
)
"""Actions the minor lobe must NEVER perform; they always escalate to the
primary lobe (or a human reviewer), regardless of any other conditions."""

# ---------------------------------------------------------------------------
# Escalation conditions
# ---------------------------------------------------------------------------

ESCALATION_CONDITIONS: frozenset[str] = frozenset(
    {
        "needs_codebase_context",
        "security_sensitive",
        "architectural_decision",
        "write_or_delete_operation",
        "final_review_required",
    }
)
"""Runtime signals that force escalation even when the duty is allowed.

Any *single* matching condition is sufficient to escalate.
"""

# ---------------------------------------------------------------------------
# Decision dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """Immutable result returned by :func:`decide`.

    Attributes
    ----------
    escalate:
        ``True`` if the request must be forwarded to the primary lobe (or a
        human); ``False`` if the minor lobe may handle it locally.
    reason:
        Human-readable explanation of why the decision was reached.
    matched_conditions:
        The subset of the supplied *conditions* that triggered escalation.
        Empty when *escalate* is ``False`` or when escalation was caused solely
        by a forbidden action with no matching escalation conditions.
    """

    escalate: bool
    reason: str
    matched_conditions: tuple[str, ...]


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------


def decide(
    *,
    duty: str | None = None,
    conditions: Iterable[str] = (),
) -> Decision:
    """Evaluate a proposed action against the minor-role governance policy.

    The function is **fail-closed**: any ambiguity (unknown duty, unrecognised
    condition) results in escalation rather than local handling.

    Parameters
    ----------
    duty:
        The action or duty the minor lobe is about to perform (e.g.
        ``"summarize"``).  ``None`` means no specific duty — treated as an
        allowed no-op.
    conditions:
        Zero or more runtime signals (strings) that describe the current
        request context.  Only values in :data:`ESCALATION_CONDITIONS` are
        meaningful; unrecognised strings are ignored (they do not escalate by
        themselves but also do not suppress escalation from known conditions).

    Returns
    -------
    Decision
        A frozen dataclass with ``escalate``, ``reason``, and
        ``matched_conditions`` fields.

    Examples
    --------
    >>> decide(duty="summarize")
    Decision(escalate=False, reason='Duty is allowed; no escalation conditions.', ...)

    >>> decide(duty="approve").escalate
    True

    >>> decide(duty="classify", conditions=["security_sensitive"]).escalate
    True
    """
    conditions_seq: tuple[str, ...] = tuple(conditions)

    # -- Rule 1: forbidden action → always escalate -------------------------
    if duty is not None and duty in FORBIDDEN:
        return Decision(
            escalate=True,
            reason=f"Forbidden action {duty!r}; minor role may not perform this.",
            matched_conditions=(),
        )

    # -- Rule 2: any recognised escalation condition present → escalate ------
    matched: tuple[str, ...] = tuple(c for c in conditions_seq if c in ESCALATION_CONDITIONS)
    if matched:
        joined = ", ".join(matched)
        return Decision(
            escalate=True,
            reason=f"Escalation condition(s) present: {joined}.",
            matched_conditions=matched,
        )

    # -- Rule 3: unknown duty (not allowed, not forbidden) → escalate --------
    if duty is not None and duty not in ALLOWED:
        return Decision(
            escalate=True,
            reason=(
                f"Unknown duty {duty!r}; not in the minor-role allowed list. "
                "Fail-closed: escalating."
            ),
            matched_conditions=(),
        )

    # -- Rule 4: allowed duty (or no duty), no escalation conditions ----------
    return Decision(
        escalate=False,
        reason="Duty is allowed; no escalation conditions.",
        matched_conditions=(),
    )
