"""Tests for model-gear's introspection verbs: overview, cli overview, doctor."""

from __future__ import annotations

import json

import pytest

from model_gear.cli import main

# --- overview -------------------------------------------------------------


def test_overview_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["overview"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# model-gear" in out
    assert "Currently served" in out
    assert "Verbs" in out


def test_overview_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "model-gear"
    assert isinstance(payload["sections"], list)
    assert payload["sections"]


def test_overview_current_only(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["overview", "--current", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    titles = [s["title"] for s in payload["sections"]]
    assert titles == ["Currently served"]


def test_overview_list_only(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["overview", "--list", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    titles = [s["title"] for s in payload["sections"]]
    assert titles == ["Supported models"]


def test_overview_graceful_on_bad_path(capsys: pytest.CaptureFixture[str]) -> None:
    # Rubric contract: descriptive verbs never hard-fail on a missing target.
    rc = main(["overview", "/no/such/path/here"])
    assert rc == 0
    assert capsys.readouterr().out.strip()


# --- cli overview ---------------------------------------------------------


def test_cli_overview_text(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cli", "overview"])
    assert rc == 0
    assert "# model cli" in capsys.readouterr().out


def test_cli_overview_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cli", "overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["subject"] == "model cli"
    assert isinstance(payload["sections"], list)


def test_cli_noun_bare_is_non_empty(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["cli"])
    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_cli_overview_unknown_flag_structured_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # `cli overview` parse errors must route through the structured error
    # contract (error:/hint: + exit 1), not argparse's default stderr/exit 2.
    with pytest.raises(SystemExit) as exc:
        main(["cli", "overview", "--bogus"])
    assert exc.value.code == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert "hint:" in err


# --- doctor ---------------------------------------------------------------


def test_doctor_text(capsys: pytest.CaptureFixture[str]) -> None:
    # Offline fixture → docker unavailable → unhealthy → exit 1.
    rc = main(["doctor"])
    assert rc == 1
    assert "model doctor" in capsys.readouterr().out


def test_doctor_json_shape(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["doctor", "--json"])
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload["healthy"], bool)
    assert isinstance(payload["checks"], list)
    assert payload["checks"]
    for check in payload["checks"]:
        assert {"id", "passed", "severity", "message", "remediation"} <= set(check)
