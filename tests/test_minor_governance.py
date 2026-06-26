"""Tests for lobes.minor.governance — role-keyed governance + escalation model.

Covers:
- An allowed duty with no escalation conditions → handled locally (escalate=False).
- A forbidden action → always escalates, regardless of conditions.
- Each of the five escalation conditions individually → escalates.
- The governance module is role-keyed (no model id in governance.py).
"""

from __future__ import annotations

import inspect

import pytest

from lobes.minor.governance import (
    ALLOWED,
    ESCALATION_CONDITIONS,
    FORBIDDEN,
    ROLE,
    Decision,
    decide,
)

# ---------------------------------------------------------------------------
# Sanity: the module is keyed to the *role*, not a model id
# ---------------------------------------------------------------------------


def test_role_constant_is_minor():
    """ROLE must be the string 'minor', not a model identifier."""
    assert ROLE == "minor"


def test_governance_module_has_no_model_id():
    """governance.py must not contain any model identifier strings.

    This ensures the policy stays role-keyed so swapping the underlying model
    is purely a catalog change with no governance edits.  We look for the
    known model-id fragments used by the minor backend.
    """
    import lobes.minor.governance as mod

    source_path = inspect.getfile(mod)
    text = open(source_path).read()

    # Known model-id fragments to guard against (add more if the catalog grows)
    forbidden_fragments = [
        "Qwen",
        "qwen",
        "4B",
        "0.6B",
        "mini",
        "smol",
        "phi",
        "gemma",
        "llama",
        "mistral",
    ]
    for fragment in forbidden_fragments:
        assert fragment not in text, (
            f"governance.py must not reference model id fragment {fragment!r}; "
            "governance is role-keyed, not model-keyed."
        )


# ---------------------------------------------------------------------------
# Catalog completeness
# ---------------------------------------------------------------------------


def test_allowed_duties_set():
    expected = {
        "prepare",
        "classify",
        "format",
        "validate",
        "suggest",
        "summarize",
        "route",
    }
    assert expected == ALLOWED


def test_forbidden_actions_set():
    expected = {
        "approve",
        "finalize",
        "delete",
        "deploy",
        "architectural_decision",
    }
    assert expected == FORBIDDEN


def test_escalation_conditions_set():
    expected = {
        "needs_codebase_context",
        "security_sensitive",
        "architectural_decision",
        "write_or_delete_operation",
        "final_review_required",
    }
    assert expected == ESCALATION_CONDITIONS


# ---------------------------------------------------------------------------
# decide() — allowed duty, no conditions → handled locally
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "duty",
    sorted(
        ALLOWED
        if False
        else ["prepare", "classify", "format", "validate", "suggest", "summarize", "route"]
    ),
)
def test_allowed_duty_no_conditions_is_handled(duty):
    """Every allowed duty with zero conditions is handled locally."""
    result = decide(duty=duty, conditions=())
    assert isinstance(result, Decision)
    assert result.escalate is False, f"duty={duty!r} should be handled locally"
    assert result.reason  # non-empty reason string
    assert result.matched_conditions == ()


# ---------------------------------------------------------------------------
# decide() — forbidden action ALWAYS escalates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action", ["approve", "finalize", "delete", "deploy", "architectural_decision"]
)
def test_forbidden_action_always_escalates(action):
    """Forbidden actions escalate regardless of supplied conditions."""
    # Without extra conditions
    result = decide(duty=action, conditions=())
    assert result.escalate is True, f"forbidden action {action!r} must escalate"
    assert result.reason


@pytest.mark.parametrize(
    "action", ["approve", "finalize", "delete", "deploy", "architectural_decision"]
)
def test_forbidden_action_escalates_even_without_extra_conditions(action):
    """Supplying an allowed duty alongside a forbidden one is not meaningful,
    but a forbidden *duty* still escalates unconditionally."""
    result = decide(duty=action, conditions=("needs_codebase_context",))
    assert result.escalate is True


# ---------------------------------------------------------------------------
# decide() — each escalation condition escalates (allowed duty)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "condition",
    [
        "needs_codebase_context",
        "security_sensitive",
        "architectural_decision",
        "write_or_delete_operation",
        "final_review_required",
    ],
)
def test_each_escalation_condition_triggers_escalation(condition):
    """Each escalation condition independently forces escalation even when
    the duty is allowed."""
    result = decide(duty="summarize", conditions=(condition,))
    assert result.escalate is True, f"condition {condition!r} should trigger escalation"
    assert condition in result.matched_conditions


# ---------------------------------------------------------------------------
# decide() — multiple conditions: matched_conditions includes all
# ---------------------------------------------------------------------------


def test_multiple_conditions_all_matched():
    result = decide(
        duty="classify",
        conditions=("needs_codebase_context", "security_sensitive"),
    )
    assert result.escalate is True
    assert "needs_codebase_context" in result.matched_conditions
    assert "security_sensitive" in result.matched_conditions


# ---------------------------------------------------------------------------
# decide() — unrecognised duty (neither allowed nor forbidden) escalates
# ---------------------------------------------------------------------------


def test_unknown_duty_escalates():
    """A duty not listed in ALLOWED escalates by default (fail-closed)."""
    result = decide(duty="unknown_verb", conditions=())
    assert result.escalate is True
    assert result.reason


# ---------------------------------------------------------------------------
# decide() — no duty supplied, no conditions → handled locally
# ---------------------------------------------------------------------------


def test_no_duty_no_conditions_is_handled():
    """Calling decide with nothing supplied is a no-op: handled locally."""
    result = decide()
    assert result.escalate is False


# ---------------------------------------------------------------------------
# Decision dataclass shape
# ---------------------------------------------------------------------------


def test_decision_has_required_fields():
    result = decide(duty="format")
    assert hasattr(result, "escalate")
    assert hasattr(result, "reason")
    assert hasattr(result, "matched_conditions")
    assert isinstance(result.escalate, bool)
    assert isinstance(result.reason, str)
    assert isinstance(result.matched_conditions, tuple)
