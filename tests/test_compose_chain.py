"""The single compose ``-f`` chain authority (issue #137).

Four hand-rolled ``-f`` builders once decided which compose files a deployment
is made of, and they drifted (#135/#136 found the same bug in two of them; the
#137 sweep found three more sites). The fix is ONE composition authority —
:func:`lobes.runtime._compose.compose_file_args` — with every caller routed
through it. These tests are the drift-proofing the #136 lesson asked for: every
caller must resolve the IDENTICAL file set for the same deployment, across the
full overlay matrix, so a fifth builder can never quietly disagree again.
"""

from __future__ import annotations

import json
import types

import pytest

from lobes.cli import main
from lobes.cli._commands import up as up_cmd
from lobes.runtime import _compose

# The overlay matrix: every combination of the two lobes-authored overlays,
# each with and without the operator's own docker-compose.override.yml.
MATRIX = [
    pytest.param(audio, shape, local, id=f"audio={audio}-shape={shape}-override={local}")
    for audio in (False, True)
    for shape in (False, True)
    for local in (False, True)
]


def _deployment(tmp_path, *, audio: bool, shape: bool, local: bool):
    """A minimal deployment dir with exactly the requested overlay files."""
    (tmp_path / _compose.COMPOSE_FILE).write_text("services: {}\n", encoding="utf-8")
    if audio:
        (tmp_path / _compose.AUDIO_OVERLAY).write_text("services: {}\n", encoding="utf-8")
    if shape:
        (tmp_path / _compose.SHAPE_OVERLAY).write_text(
            'services:\n  vllm-multimodal:\n    profiles: ["shape-dropped"]\n'
            "  gateway:\n    depends_on: !reset null\n",
            encoding="utf-8",
        )
    if local:
        (tmp_path / _compose.LOCAL_OVERRIDE).write_text("services: {}\n", encoding="utf-8")
    return tmp_path


def _expected(*, audio: bool, shape: bool, local: bool) -> list[str]:
    """The chain the authority is expected to emit — spelled out once, here."""
    if not audio and not shape:
        return []
    files = ["-f", _compose.COMPOSE_FILE]
    if audio:
        files += ["-f", _compose.AUDIO_OVERLAY]
    if shape:
        files += ["-f", _compose.SHAPE_OVERLAY]
    if local:
        files += ["-f", _compose.LOCAL_OVERRIDE]
    return files


def _record_runs(monkeypatch) -> list[list[str]]:
    """Capture every argv ``_compose._run`` would execute."""
    calls: list[list[str]] = []

    def fake_run(argv, **kwargs):
        calls.append(list(argv))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_compose, "_run", fake_run)
    return calls


@pytest.mark.parametrize("audio,shape,local", MATRIX)
def test_every_caller_resolves_the_identical_file_set(
    tmp_path, monkeypatch, capsys, audio, shape, local
) -> None:
    """The equivalence gate: builder, up.py delegation, every compose verb, and
    the script-facing ``lobes fleet files`` all agree, byte for byte."""
    deploy = _deployment(tmp_path, audio=audio, shape=shape, local=local)
    expected = _expected(audio=audio, shape=shape, local=local)

    # 1. The authority itself, via the dir-probing wrapper.
    assert _compose._compose_files(deploy) == expected

    # 2. The pure form the role-targeted `lobes up` uses (audio forced to the
    #    overlay's presence makes its semantics coincide with the default).
    assert _compose.compose_file_args(audio=audio, shape=shape, local=local) == expected
    assert up_cmd._compose_file_args(audio, shape, local) == expected

    # 3. Every mutating compose verb: down and both ups carry the same chain.
    calls = _record_runs(monkeypatch)
    _compose.compose_down(deploy)
    _compose.compose_up_detached(deploy)
    _compose.compose_up_build(deploy)
    down_argv, up_argv, build_argv = calls
    assert down_argv == ["docker", "compose"] + expected + ["down"]
    assert up_argv == ["docker", "compose"] + expected + ["up", "-d"]
    assert build_argv == ["docker", "compose"] + expected + ["up", "-d", "--build"]

    # 4. The script-facing read verb emits the same chain, one token per line
    #    (nothing at all for a plain deployment — bash mapfile gets []).
    rc = main(["fleet", "files", "--compose-dir", str(deploy)])
    assert rc == 0
    out = capsys.readouterr().out
    assert [line for line in out.splitlines() if line] == expected


@pytest.mark.parametrize("audio,shape,local", MATRIX)
def test_fleet_files_json_matches_the_authority(tmp_path, capsys, audio, shape, local) -> None:
    deploy = _deployment(tmp_path, audio=audio, shape=shape, local=local)
    rc = main(["fleet", "files", "--compose-dir", str(deploy), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["files"] == _expected(audio=audio, shape=shape, local=local)
    assert payload["deployment_dir"] == str(deploy)


def test_up_and_down_agree_on_a_shape_deployment(tmp_path, monkeypatch) -> None:
    """The #137 asymmetry, pinned: ``switch``/``serve`` used to tear down with
    the full chain and bring up with NONE — on a shape deployment the bare
    ``docker compose up -d`` skipped the shape override, so the dropped lobe
    booted and ate the GPU budget the shape reclaimed (proven live on the
    GB10). Up and down must resolve the same file set, always."""
    deploy = _deployment(tmp_path, audio=False, shape=True, local=True)
    calls = _record_runs(monkeypatch)
    _compose.compose_down(deploy)
    _compose.compose_up_detached(deploy)
    down_argv, up_argv = calls
    assert down_argv[2:-1] == up_argv[2:-2]  # identical -f chain either way
    assert _compose.SHAPE_OVERLAY in up_argv  # the dropped lobe stays parked


def test_plain_deployment_keeps_the_bare_argv(tmp_path, monkeypatch) -> None:
    """No lobes overlay ⇒ NO ``-f`` at all, even with an operator override on
    disk: compose resolves the project itself and its own convention layers
    base + docker-compose.override.yml. Passing ``-f`` here would change
    nothing except to break that convention (the c16 boundary)."""
    deploy = _deployment(tmp_path, audio=False, shape=False, local=True)
    assert _compose._compose_files(deploy) == []
    calls = _record_runs(monkeypatch)
    _compose.compose_up_detached(deploy)
    _compose.compose_down(deploy)
    assert calls[0] == ["docker", "compose", "up", "-d"]
    assert calls[1] == ["docker", "compose", "down"]


def test_targeted_audio_semantics_still_differ_by_design(tmp_path) -> None:
    """`lobes up <role>`'s "only the overlays the TARGETED services need": on an
    audio-scaffolded deployment, a non-audio target excludes the audio overlay
    — the one semantic the parameterisation preserves from the old parallel
    builder (it is a parameter now, not a second copy)."""
    deploy = _deployment(tmp_path, audio=True, shape=True, local=False)
    assert _compose._compose_files(deploy, audio=False) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.SHAPE_OVERLAY,
    ]
    assert _compose._compose_files(deploy, audio=True) == _compose._compose_files(deploy)
