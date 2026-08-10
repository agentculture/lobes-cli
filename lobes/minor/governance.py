"""Governance and escalation model for the **hand** role (the cheap tier).

This module encodes *what* the cheap-tier lobe MAY do vs MUST escalate, keyed
entirely to the *role name* — never to a specific model identifier. Swapping the
underlying model is a catalog-only change; nothing here needs to change.

That claim was TESTED and held when the `hand` lobe took over this tier: the
model swap itself needed nothing here. What DID change is the role NAME —
`minor` is now a back-compat tier spelling, not a role — so :data:`ROLE` reads
``"hand"``. The duty lists below were re-derived for an adapter-dependent
specialist and came out unchanged; see :data:`ALLOWED` for why that is a
decision and not an oversight. (Which checkpoint `hand` serves is deliberately
not stated anywhere in this module — see :data:`ROLE`.)

The module keeps its ``lobes.minor`` package path. That is deliberate
cite-don't-delete: ``lobes run minor``, ``lobes route`` and ``lobes.bench`` all
import from it, and renaming the package would break every one of them to
express a fact the ``ROLE`` constant already states.

Public API
----------
ROLE : str
    The role name this governance policy applies to (``"hand"``).
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

ROLE: str = "hand"
"""The role name this governance policy applies to.

Governance is role-keyed, not model-keyed, so that the underlying model can be
swapped in the catalog without touching this file.

``"hand"``, not ``"minor"``: the `hand` lobe took over the cheap tier, and
``minor``/``cheap`` survive only as back-compat TIER spellings that resolve to
this same role (see :data:`lobes.catalog.TIER_ROLE`). Leaving this at
``"minor"`` would have left exactly one authority disagreeing with every other
surface about what governs this lane.

The catalog owns which checkpoint this role serves; this module never names
one, and ``tests/test_minor_governance.py`` asserts that.
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
"""Duties the cheap-tier lobe may perform locally (no escalation required).

RE-DERIVED for an adapter-dependent specialist and deliberately UNCHANGED.
`hand` is a trained specialist — a loaded LoRA adapter is meant to make it
genuinely good at one domain — and the tempting move is to widen this set when
an adapter is present. That is refused, for two reasons:

* **Competence is not authority.** An adapter makes `hand` better at the duties
  it already has; it does not grant it new ones. A legal-trained 1.2B that can
  draft a clause well is still not the lobe that APPROVES one.
* **Governance must be decidable without asking the engine.** Keying the
  allowed set to which adapters happen to be loaded would make the same duty
  legal on one box and forbidden on another, and would make this pure stdlib
  module depend on live serving state.

So the policy is flat: every `hand` request is governed identically, base or
adapter. Widening a duty later is contract-compatible; narrowing one is a
break — the same asymmetry that keeps ``repo_action`` in the role's forbidden
list for v1 (see ``lobes.roles.ROLE_FORBIDDEN``, issue #180).
"""

FORBIDDEN: frozenset[str] = frozenset(
    {
        "approve",
        "finalize",
        "delete",
        "deploy",
        "architectural_decision",
    }
)
"""Actions the cheap-tier lobe must NEVER perform; they always escalate to the
primary lobe (or a human reviewer), regardless of any other conditions.

These are the DUTY-level spelling of ``ROLE_FORBIDDEN["hand"]`` in
``lobes.roles`` (``final_decision`` / ``repo_action`` / ``security_decision``):
``approve``/``finalize`` are the final decision, ``delete``/``deploy`` are repo
and infrastructure actions, and ``architectural_decision`` is both. The two
lists are different vocabularies — a runtime duty check here, a Colleague-facing
contract there — describing one boundary, and neither may quietly outgrow the
other."""

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
# Uncertainty threshold
# ---------------------------------------------------------------------------

UNCERTAINTY_THRESHOLD: float = 0.25
"""Confidence floor below which a routing/classification decision escalates.

Mirrors issue #64's ``escalation.uncertainty_threshold``: when a caller supplies
a ``confidence`` to :func:`decide` and it falls *below* this value, the
cheap-tier lobe is too unsure to handle the task locally and the decision
escalates.
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
    confidence: float | None = None,
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
    confidence:
        Optional self-reported confidence in ``[0, 1]``. When supplied and below
        :data:`UNCERTAINTY_THRESHOLD`, the decision escalates (the lobe is too
        unsure to handle the task locally).

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

    # -- Rule 2b: confidence below the uncertainty threshold → escalate ------
    if confidence is not None and confidence < UNCERTAINTY_THRESHOLD:
        return Decision(
            escalate=True,
            reason=(
                f"Low confidence {confidence:.2f} < uncertainty threshold "
                f"{UNCERTAINTY_THRESHOLD}; escalating."
            ),
            matched_conditions=(),
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
