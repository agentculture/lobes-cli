"""Tests for ``lobes init --fleet --audio --audio-lang he`` (hebrew-realtime t15).

Language is an OVERLAY choice, not a shape and not a variation (frame c16): the
Hebrew audio overlay layers on top of the English one — the file chain is
``docker-compose.yml`` -> ``docker-compose.audio.yml`` ->
``docker-compose.audio-he.yml`` — so every assertion here is about a THIRD
layer being added, never about the English layer changing.

The load-bearing half of this file is the negative: ``--audio`` with no
``--audio-lang`` (and ``--audio-lang en``, its explicit spelling) must scaffold
exactly what it scaffolded before this task existed, byte for byte.
"""

from __future__ import annotations

import json
import re

import pytest

from lobes.cli import main
from lobes.runtime import _compose, _detect


@pytest.fixture(autouse=True)
def _pin_spark_detection(monkeypatch) -> None:
    """Pin detection to the GB10 — host-independence, mirroring tests/test_init.py."""
    card = _detect.DetectedCard(
        resolved="spark",
        device_name="NVIDIA GB10",
        compute_capability="sm_121",
        total_memory_gb=119.7,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )
    monkeypatch.setattr(_detect, "detect_card", lambda: card)


# --- criterion 1a: the Hebrew selection materialises the third layer --------


def test_audio_lang_he_dry_run_json_lists_the_hebrew_overlay(tmp_path, capsys) -> None:
    target = tmp_path / "he"
    rc = main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audio"] is True
    assert payload["audio_lang"] == "he"
    names = {f["name"] for f in payload["files"]}
    # Derive from the source map so a dropped/added template can't slip past.
    assert set(_compose.AUDIO_TEMPLATES.values()) <= names
    assert set(_compose.AUDIO_HE_TEMPLATES.values()) <= names
    assert not target.exists()


def test_audio_lang_he_dry_run_text_mentions_the_hebrew_env_append(tmp_path, capsys) -> None:
    target = tmp_path / "he"
    rc = main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target)])
    assert rc == 0
    out = capsys.readouterr().out
    assert _compose.AUDIO_HE_OVERLAY in out
    assert ".env (+ audio keys appended, he)" in out
    assert "Re-run with --apply to write." in out
    assert not target.exists()


def test_audio_lang_he_apply_writes_every_hebrew_file(tmp_path) -> None:
    target = tmp_path / "he"
    assert main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target), "--apply"]) == 0
    for dest in _compose.AUDIO_TEMPLATES.values():
        assert (target / dest).is_file(), dest
    for dest in _compose.AUDIO_HE_TEMPLATES.values():
        assert (target / dest).is_file(), dest
    # The Whisper sidecar's server module is COPY'd by Dockerfile.whisper-stt, so
    # it must land at the deployment root exactly like listen_server.py does for
    # Parakeet — otherwise `docker compose build stt` fails on the COPY.
    assert (target / "listen_server_whisper.py").is_file()


def test_audio_lang_he_apply_appends_hebrew_env_after_english(tmp_path) -> None:
    target = tmp_path / "he"
    assert main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target), "--apply"]) == 0
    env = (target / _compose.ENV_FILE).read_text(encoding="utf-8")
    # Base fleet keys, then the English audio keys, then the Hebrew ones — in
    # that order, because .env is append-only and last-wins for compose.
    assert env.index("PRIMARY_MODEL=") < env.index("CHATTERBOX_PORT=")
    assert env.index("CHATTERBOX_PORT=") < env.index("REALTIME_LANGUAGE=he")
    assert "STT_MODEL=ivrit-ai/whisper-large-v3-turbo" in env
    assert "TTS_RUNTIME=chatterbox-multilingual" in env


def test_audio_lang_he_apply_never_rewrites_an_existing_env_line(tmp_path) -> None:
    """The append-only guarantee (#174) holds for the Hebrew keys too: every
    line the English pass wrote is still present, verbatim, afterwards."""
    target = tmp_path / "he"
    assert main(["init", "--fleet", "--audio", str(target), "--apply"]) == 0
    english = (target / _compose.ENV_FILE).read_text(encoding="utf-8")
    target_he = tmp_path / "he2"
    assert (
        main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target_he), "--apply"]) == 0
    )
    hebrew = (target_he / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert hebrew.startswith(english)


def test_audio_lang_he_reapply_does_not_duplicate_or_clobber_an_edited_key(tmp_path) -> None:
    """Qodo finding: a re-run of `init --fleet --audio --audio-lang he --apply`
    used to APPEND the whole Hebrew template again with no de-duplication, so
    after an operator set PHONIKUD_MODEL_PATH, a re-run appended a second,
    blank PHONIKUD_MODEL_PATH= line — and since docker compose `env_file`
    semantics make the LAST occurrence win, the operator's value was silently
    lost. This must survive a re-run with exactly one occurrence, holding the
    operator's edited value."""
    target = tmp_path / "he"
    assert main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target), "--apply"]) == 0
    env_path = target / _compose.ENV_FILE
    original = env_path.read_text(encoding="utf-8")
    assert "PHONIKUD_MODEL_PATH=" in original
    edited = original.replace(
        "PHONIKUD_MODEL_PATH=\n", "PHONIKUD_MODEL_PATH=/opt/models/phonikud\n"
    )
    assert edited != original
    env_path.write_text(edited, encoding="utf-8")

    assert main(["init", "--fleet", "--audio", "--audio-lang", "he", str(target), "--apply"]) == 0

    reapplied = env_path.read_text(encoding="utf-8")
    lines = reapplied.splitlines()
    phonikud_lines = [ln for ln in lines if ln.startswith("PHONIKUD_MODEL_PATH=")]
    assert len(phonikud_lines) == 1, phonikud_lines
    assert phonikud_lines[0] == "PHONIKUD_MODEL_PATH=/opt/models/phonikud"


def test_every_hebrew_compose_referenced_dockerfile_is_scaffolded(tmp_path) -> None:
    """Same guardrail as tests/test_init.py's English version: a `dockerfile:`
    the Hebrew overlay references MUST itself be scaffolded."""
    templates = {
        **_compose.FLEET_TEMPLATES,
        **_compose.AUDIO_TEMPLATES,
        **_compose.AUDIO_HE_TEMPLATES,
    }
    _compose.write_scaffold(tmp_path, force=True, templates=templates)
    text = (tmp_path / _compose.AUDIO_HE_OVERLAY).read_text(encoding="utf-8")
    refs = re.findall(r"^\s*dockerfile:\s*(\S+)", text, re.MULTILINE)
    assert refs, "the Hebrew overlay is expected to override at least one build"
    for ref in refs:
        assert (tmp_path / ref).is_file(), f"{_compose.AUDIO_HE_OVERLAY} builds from {ref}"


# --- criterion 1b: BlueTTS ships, unwired (approved deviation d8) -----------


def test_bluetts_dockerfile_ships_but_is_wired_into_no_compose_file(tmp_path) -> None:
    """Dockerfile.bluetts is materialised for `he` so an operator can opt in,
    but nothing wires it: the shipped Hebrew default voice stays Chatterbox
    Multilingual, and its weights repo has no declared licence."""
    assert "Dockerfile.bluetts" in _compose.AUDIO_HE_TEMPLATES.values()
    templates = {
        **_compose.FLEET_TEMPLATES,
        **_compose.AUDIO_TEMPLATES,
        **_compose.AUDIO_HE_TEMPLATES,
    }
    _compose.write_scaffold(tmp_path, force=True, templates=templates)
    assert (tmp_path / "Dockerfile.bluetts").is_file()
    for compose_name in (_compose.COMPOSE_FILE, _compose.AUDIO_OVERLAY, _compose.AUDIO_HE_OVERLAY):
        text = (tmp_path / compose_name).read_text(encoding="utf-8")
        assert "Dockerfile.bluetts" not in text, compose_name
    # No default anywhere names a BlueTTS weights repo — the operator fetches
    # them deliberately (hard operator constraint, deviation d8).
    env = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert "BLUETTS_WEIGHTS_REPO" not in env


# --- criterion 1c: without the flag, nothing moves --------------------------


def test_plain_audio_is_english_and_writes_no_hebrew_file(tmp_path, capsys) -> None:
    target = tmp_path / "en"
    rc = main(["init", "--fleet", "--audio", str(target), "--json", "--apply"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["audio"] is True
    assert payload["audio_lang"] == "en"
    written = {p.name for p in target.iterdir() if p.is_file()}
    assert written.isdisjoint(set(_compose.AUDIO_HE_TEMPLATES.values()))
    assert "REALTIME_LANGUAGE" not in (target / _compose.ENV_FILE).read_text(encoding="utf-8")


def test_audio_lang_en_scaffolds_byte_identically_to_plain_audio(tmp_path) -> None:
    """``--audio-lang en`` is EXACTLY today's ``--audio``, byte for byte."""
    plain = tmp_path / "plain"
    explicit = tmp_path / "explicit"
    assert main(["init", "--fleet", "--audio", str(plain), "--apply"]) == 0
    assert main(["init", "--fleet", "--audio", "--audio-lang", "en", str(explicit), "--apply"]) == 0
    a = sorted(p.name for p in plain.iterdir() if p.is_file())
    b = sorted(p.name for p in explicit.iterdir() if p.is_file())
    assert a == b
    for name in a:
        assert (plain / name).read_bytes() == (explicit / name).read_bytes(), name


# --- criterion 1d: the flag's error surface ---------------------------------


def test_audio_lang_he_without_audio_is_a_user_error(tmp_path, capsys) -> None:
    rc = main(["init", "--fleet", "--audio-lang", "he", str(tmp_path / "x")])
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "--audio-lang" in err
    assert "--audio" in err


def test_audio_lang_unknown_is_a_user_error(tmp_path, capsys) -> None:
    rc = main(["init", "--fleet", "--audio", "--audio-lang", "fr", str(tmp_path / "x")])
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "fr" in err
    assert "he" in err  # the hint names the languages that exist


def test_audio_lang_he_is_incompatible_with_single(tmp_path, capsys) -> None:
    rc = main(["init", "--single", "--audio-lang", "he", str(tmp_path / "x")])
    assert rc == 1
    assert "--single" in capsys.readouterr().err


def test_audio_lang_he_is_incompatible_with_from_lock(tmp_path, capsys) -> None:
    rc = main(["init", "--from-lock", str(tmp_path), "--audio-lang", "he"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--from-lock" in err
    assert "--audio-lang" in err
