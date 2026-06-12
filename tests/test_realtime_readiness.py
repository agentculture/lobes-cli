"""Tests for the Parakeet readiness decision logic (stdlib + pytest only).

No torch / nemo / fastapi imports — the module under test is stdlib-only so
these tests run in the offline CI environment (no GPU, no container deps).
Mirrors the style of tests/test_realtime_settings.py.
"""

from __future__ import annotations

import pytest

from model_gear.realtime._readiness import evaluate_readiness


def test_both_ok_returns_200_ready() -> None:
    code, body = evaluate_readiness(model_loaded=True, cuda_ok=True)
    assert code == 200
    assert body == {"status": "ready"}


def test_model_not_loaded_returns_503_with_reason() -> None:
    code, body = evaluate_readiness(model_loaded=False, cuda_ok=True)
    assert code == 503
    assert body["status"] == "not_ready"
    assert "model" in body["reason"].lower()


def test_cuda_not_available_returns_503_with_cuda_in_reason() -> None:
    code, body = evaluate_readiness(model_loaded=True, cuda_ok=False)
    assert code == 503
    assert body["status"] == "not_ready"
    assert "cuda" in body["reason"].lower()


def test_neither_loaded_reports_503_model_reason_first() -> None:
    """When both flags are False the model check fires first (load order)."""
    code, body = evaluate_readiness(model_loaded=False, cuda_ok=False)
    assert code == 503
    assert body["status"] == "not_ready"
    assert "model" in body["reason"].lower()


@pytest.mark.parametrize(
    "model_loaded, cuda_ok, expected_code",
    [
        (True, True, 200),
        (True, False, 503),
        (False, True, 503),
        (False, False, 503),
    ],
)
def test_status_code_matrix(model_loaded: bool, cuda_ok: bool, expected_code: int) -> None:
    code, _body = evaluate_readiness(model_loaded=model_loaded, cuda_ok=cuda_ok)
    assert code == expected_code
