"""Smoke tests for the model-gear CLI entry point and the agent-first verbs."""

from __future__ import annotations

import json

import pytest

from model_gear import __version__
from model_gear.cli import main
from model_gear.explain import known_paths


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    assert "usage: model" in capsys.readouterr().out


def test_unknown_command_errors(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["bogus"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


# --- whoami ---------------------------------------------------------------


def test_whoami_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "tool: model-gear" in out
    assert "served_model:" in out
    assert "container_health:" in out
    assert "agent: model-gear" in out


def test_whoami_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "model-gear"
    assert payload["version"] == __version__
    assert isinstance(payload["machine"], dict)
    assert "host" in payload["machine"]
    assert payload["served_model"]
    assert payload["agent"] == "model-gear"
    # Offline fixture → no container.
    assert payload["container_health"] == "not created"


# --- learn ----------------------------------------------------------------


def test_learn_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "model-gear" in out
    assert "Exit-code policy" in out
    assert "--json" in out
    assert "switch" in out
    assert "Mutation safety" in out


def test_learn_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "model-gear"
    assert payload["version"] == __version__
    assert payload["json_support"] is True
    assert payload["serves"] == "model-gear"
    verbs = {tuple(c["path"]) for c in payload["commands"]}
    assert ("switch",) in verbs
    assert ("assess",) in verbs
    assert ("fleet",) in verbs
    assert set(payload["mutation_safety"]["write_verbs"]) == {
        "switch",
        "serve",
        "stop",
        "init",
        "fleet up",
        "fleet down",
    }


# --- explain --------------------------------------------------------------


def test_explain_root(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# model-gear" in out
    assert "switch" in out


def test_explain_self_uses_prog_name(capsys: pytest.CaptureFixture[str]) -> None:
    # The rubric probes `explain model` (the binary name) — must resolve to root.
    rc = main(["explain", "model"])
    assert rc == 0
    assert "# model-gear" in capsys.readouterr().out


def test_explain_switch(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "switch"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out.lower()
    assert "--apply" in out


def test_explain_backend(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "backend"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "acp" in out
    assert "vllm-local/" in out
    assert "model-gear-vllm" in out


def test_explain_models(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "models"])
    assert rc == 0
    assert "docs/" in capsys.readouterr().out


def test_explain_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "switch", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == ["switch"]
    assert "model switch" in payload["markdown"]


def test_explain_unknown_path_errors(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "nonexistent"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "hint:" in captured.err


def test_every_catalog_path_resolves(capsys: pytest.CaptureFixture[str]) -> None:
    for path in known_paths():
        rc = main(["explain", *path])
        assert rc == 0, f"explain {' '.join(path)} failed"
        capsys.readouterr()
