"""Smoke tests for the lobes CLI entry point and the agent-first verbs."""

from __future__ import annotations

import json

import pytest

from lobes import __version__
from lobes.cli import main
from lobes.explain import known_paths


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_args_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    assert rc == 0
    assert "usage: lobes" in capsys.readouterr().out


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
    assert "tool: lobes" in out
    assert "served_model:" in out
    assert "container_health:" in out
    assert "agent: lobes" in out


def test_whoami_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["whoami", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "lobes"
    assert payload["version"] == __version__
    assert isinstance(payload["machine"], dict)
    assert "host" in payload["machine"]
    assert payload["served_model"]
    assert payload["agent"] == "lobes"
    # Offline fixture → no container.
    assert payload["container_health"] == "not created"


# --- learn ----------------------------------------------------------------


def test_learn_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "lobes" in out
    assert "Exit-code policy" in out
    assert "--json" in out
    assert "switch" in out
    assert "Mutation safety" in out


def test_learn_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["learn", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["tool"] == "lobes"
    assert payload["version"] == __version__
    assert payload["json_support"] is True
    assert payload["serves"] == "lobes"
    verbs = {tuple(c["path"]) for c in payload["commands"]}
    assert ("switch",) in verbs
    assert ("assess",) in verbs
    assert ("fleet",) in verbs
    assert ("tunnel",) in verbs
    assert set(payload["mutation_safety"]["write_verbs"]) == {
        "switch",
        "serve",
        "stop",
        "up",
        "init",
        "fleet up",
        "fleet down",
        "tunnel",
    }


# --- explain --------------------------------------------------------------


def test_explain_root(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# lobes" in out
    assert "switch" in out


def test_explain_self_uses_prog_name(capsys: pytest.CaptureFixture[str]) -> None:
    # The rubric probes `explain model` (the binary name) — must resolve to root.
    rc = main(["explain", "model"])
    assert rc == 0
    assert "# lobes" in capsys.readouterr().out


@pytest.mark.parametrize("alias", ["lobes", "lobes-cli", "model", "model-gear"])
def test_explain_root_aliases_resolve(alias: str, capsys: pytest.CaptureFixture[str]) -> None:
    # Both the new names and the deprecated model/model-gear aliases must resolve
    # to the root entry (regression: the duplicate ("lobes",) key once dropped the
    # ("model-gear",) back-compat alias entirely).
    rc = main(["explain", alias])
    assert rc == 0
    assert "# lobes" in capsys.readouterr().out


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


@pytest.mark.parametrize("alias", ["roles", "colleague", "colleague-stack", "capabilities"])
def test_explain_roles(alias: str, capsys: pytest.CaptureFixture[str]) -> None:
    # issue #81: the six-role Colleague contract (cortex/senses/embedder/
    # reranker/stt/tts) must render under every documented alias.
    rc = main(["explain", alias])
    assert rc == 0
    out = capsys.readouterr().out
    for role in ("cortex", "senses", "embedder", "reranker", "stt", "tts"):
        assert role in out
    assert "GET /capabilities" in out
    assert "docs/colleague-stack.md" in out


def test_explain_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["explain", "switch", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["path"] == ["switch"]
    assert "lobes switch" in payload["markdown"]


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
