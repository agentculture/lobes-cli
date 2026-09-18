"""The Hebrew overlay's place in the ``docker compose -f`` chain (t15).

``docker-compose.audio-he.yml`` is a THIRD layer: it must come AFTER the
English audio overlay (it overrides that file's ``stt``/``chatterbox``/
``realtime``/``gateway`` keys by compose deep-merge, so ordering it earlier
would silently lose every Hebrew value) and BEFORE the shape override and the
operator's own ``docker-compose.override.yml`` (last wins, by convention).

It is also STRICTLY PAIRED with the English overlay — its services are
declared there, and a compose override may only name services some file in the
same chain declares — so a role-targeted ``lobes up <non-audio-role>``, which
deliberately drops the audio overlay, must drop this one too.
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.runtime import _compose


def _scaffold(path, *, hebrew: bool):
    templates = {**_compose.FLEET_TEMPLATES, **_compose.AUDIO_TEMPLATES}
    if hebrew:
        templates = {**templates, **_compose.AUDIO_HE_TEMPLATES}
    _compose.write_scaffold(path, force=True, templates=templates)


def test_chain_places_hebrew_after_audio_and_before_shape_and_override() -> None:
    files = _compose.compose_file_args(audio=True, audio_he=True, shape=True, local=True, gpu=True)
    names = [tok for tok in files if tok != "-f"]
    assert names.index(_compose.AUDIO_OVERLAY) < names.index(_compose.AUDIO_HE_OVERLAY)
    assert names.index(_compose.AUDIO_HE_OVERLAY) < names.index(_compose.SHAPE_OVERLAY)
    assert names.index(_compose.SHAPE_OVERLAY) < names.index(_compose.LOCAL_OVERRIDE)


def test_chain_never_carries_hebrew_without_the_english_overlay() -> None:
    files = _compose.compose_file_args(audio=False, audio_he=True, shape=False, local=False)
    assert _compose.AUDIO_HE_OVERLAY not in files


def test_chain_default_is_english_only() -> None:
    """Back-compat: the pre-t15 call signature renders the pre-t15 chain."""
    assert _compose.compose_file_args(audio=True, shape=False, local=False) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.AUDIO_OVERLAY,
    ]


def test_compose_files_probes_the_hebrew_overlay(tmp_path) -> None:
    _scaffold(tmp_path, hebrew=False)
    assert _compose.AUDIO_HE_OVERLAY not in _compose._compose_files(tmp_path)
    _scaffold(tmp_path, hebrew=True)
    assert _compose.AUDIO_HE_OVERLAY in _compose._compose_files(tmp_path)


def test_audio_he_overlay_present(tmp_path) -> None:
    _scaffold(tmp_path, hebrew=False)
    assert _compose.audio_he_overlay_present(tmp_path) is False
    _scaffold(tmp_path, hebrew=True)
    assert _compose.audio_he_overlay_present(tmp_path) is True


def test_fleet_files_prints_the_hebrew_overlay(tmp_path, capsys) -> None:
    _scaffold(tmp_path, hebrew=True)
    rc = main(["fleet", "files", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"] == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.AUDIO_OVERLAY,
        "-f",
        _compose.AUDIO_HE_OVERLAY,
    ]


def test_up_stt_includes_the_hebrew_overlay(tmp_path, capsys) -> None:
    _scaffold(tmp_path, hebrew=True)
    rc = main(["up", "stt", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert _compose.AUDIO_HE_OVERLAY in payload["command"]


def test_up_non_audio_role_drops_the_hebrew_overlay(tmp_path, capsys) -> None:
    _scaffold(tmp_path, hebrew=True)
    rc = main(["up", "cortex", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert _compose.AUDIO_HE_OVERLAY not in payload["command"]
    assert _compose.AUDIO_OVERLAY not in payload["command"]
