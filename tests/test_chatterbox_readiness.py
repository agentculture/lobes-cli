"""Tests for the Chatterbox TTS readiness_status helper (stdlib + pytest only).

No torch / fastapi imports — the helper is a pure function so these tests run
in the offline CI environment (no GPU, no container deps).
"""

from __future__ import annotations

from lobes.realtime.chatterbox_server import readiness_status


def test_model_loaded_and_not_poisoned_returns_200_ready() -> None:
    code, body = readiness_status(True, False)
    assert code == 200
    assert body == {"status": "ready"}


def test_model_not_loaded_returns_503_loading() -> None:
    code, body = readiness_status(False, False)
    assert code == 503
    assert body == {"status": "loading"}


def test_model_loaded_and_cuda_poisoned_returns_503_unavailable() -> None:
    code, body = readiness_status(True, True)
    assert code == 503
    assert body["status"] == "unavailable"
    assert body["reason"] == "cuda_context_poisoned"
