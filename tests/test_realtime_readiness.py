"""Tests for the Parakeet readiness decision logic (stdlib + pytest only).

No torch / nemo / fastapi imports — the module under test is stdlib-only so
these tests run in the offline CI environment (no GPU, no container deps).
Mirrors the style of tests/test_realtime_settings.py.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from lobes.realtime._readiness import evaluate_readiness

# The vendored twin COPY'd into the Parakeet image / scaffolded by
# `lobes init --fleet --audio`. It must stay behaviourally identical to the
# canonical module (cite-don't-import — two copies, one truth).
_VENDORED_TWIN = (
    Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet" / "_readiness.py"
)


def _load_vendored_evaluate_readiness():
    spec = importlib.util.spec_from_file_location("_fleet_readiness", _VENDORED_TWIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_readiness


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


@pytest.mark.parametrize(
    "model_loaded, cuda_ok",
    [(True, True), (True, False), (False, True), (False, False)],
)
def test_vendored_twin_matches_canonical(model_loaded: bool, cuda_ok: bool) -> None:
    """The fleet-template copy must not drift from the canonical decision."""
    vendored = _load_vendored_evaluate_readiness()
    assert vendored(model_loaded=model_loaded, cuda_ok=cuda_ok) == evaluate_readiness(
        model_loaded=model_loaded, cuda_ok=cuda_ok
    )
