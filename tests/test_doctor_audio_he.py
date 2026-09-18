"""``lobes doctor`` knows about the Hebrew audio overlay — and only there (t15).

The heal lane enumerates the scaffold files and ``.env`` keys a deployment is
EXPECTED to carry. The Hebrew overlay adds both, so a Hebrew box with a missing
``Dockerfile.whisper-stt`` (or a ``.env`` that predates ``STT_MODEL``) must be
flagged and healable exactly like the English overlay's own files are.

The half that matters more is the negative: an ENGLISH deployment must never be
told it is missing a Hebrew file. Hebrew is opt-in, so its absence is not a
defect — the same rule that keeps a no-audio deployment from being flagged for
lacking the audio overlay.
"""

from __future__ import annotations

import json

import pytest

from lobes.cli import main
from lobes.runtime import _compose, _detect, _env


def _card(resolved: str = "spark"):
    return _detect.DetectedCard(
        resolved=resolved,
        device_name="NVIDIA GB10",
        compute_capability="sm_121",
        total_memory_gb=128.0,
        hostname="testbox",
        device_tree_model=None,
        sources={},
    )


def _scaffold(path, *, hebrew: bool):
    templates = {**_compose.FLEET_TEMPLATES, **_compose.AUDIO_TEMPLATES}
    if hebrew:
        templates = {**templates, **_compose.AUDIO_HE_TEMPLATES}
    _compose.write_scaffold(path, force=True, templates=templates)
    _compose.write_plugin_file(path, force=True)
    _compose.append_audio_env(path)
    if hebrew:
        _compose.append_audio_he_env(path)
    _env.set_env(path / ".env", "LOBES_PROFILE", "spark")
    return path


@pytest.fixture
def _offline(monkeypatch):
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card())


def _doctor_json(capsys, tmp_path, monkeypatch) -> dict:
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    main(["doctor", "--json"])
    return json.loads(capsys.readouterr().out)


def test_hebrew_deployment_heals_its_own_missing_files(
    tmp_path, monkeypatch, capsys, _offline
) -> None:
    _scaffold(tmp_path, hebrew=True)
    (tmp_path / "Dockerfile.whisper-stt").unlink()
    (tmp_path / "listen_server_whisper.py").unlink()
    payload = _doctor_json(capsys, tmp_path, monkeypatch)
    assert set(payload["fix_plan"]["files"]) == {
        "Dockerfile.whisper-stt",
        "listen_server_whisper.py",
    }


def test_hebrew_deployment_heals_its_own_missing_env_keys(
    tmp_path, monkeypatch, capsys, _offline
) -> None:
    _scaffold(tmp_path, hebrew=True)
    env_path = tmp_path / ".env"
    kept = [
        ln
        for ln in env_path.read_text(encoding="utf-8").splitlines()
        if not ln.startswith(("STT_MODEL=", "TTS_DIACRITIZE="))
    ]
    env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    payload = _doctor_json(capsys, tmp_path, monkeypatch)
    assert payload["fix_plan"]["env"]["STT_MODEL"] == "ivrit-ai/whisper-large-v3-turbo"
    assert payload["fix_plan"]["env"]["TTS_DIACRITIZE"] == "auto"


def test_english_deployment_is_never_asked_for_a_hebrew_file(
    tmp_path, monkeypatch, capsys, _offline
) -> None:
    _scaffold(tmp_path, hebrew=False)
    payload = _doctor_json(capsys, tmp_path, monkeypatch)
    assert payload["fix_plan"]["files"] == []
    assert "STT_MODEL" not in payload["fix_plan"]["env"]
    assert "REALTIME_LANGUAGE" not in payload["fix_plan"]["env"]
