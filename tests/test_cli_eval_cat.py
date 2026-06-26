"""Tests for ``lobes eval cat`` — cat-probe logprobs eval verb (read-only).

Monkeypatching strategy: ``cmd_eval_cat`` calls ``score_case`` (imported into
the eval module's namespace as ``eval_cmd.score_case``).  Tests patch that
reference so no HTTP traffic is made.  ``generate_case`` is called for real
because it is pure/deterministic/stdlib-only.

Covers spec target: c16.
"""

from __future__ import annotations

import argparse
import json
import types

import pytest

from lobes.bench.cat_probe import generate_case
from lobes.cli._commands import eval as eval_cmd
from lobes.cli._errors import EXIT_USER_ERROR, ModelGearError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cat_args(**kwargs) -> types.SimpleNamespace:
    """Build an argparse Namespace matching what register() produces for 'eval cat'."""
    defaults: dict = {
        "json": False,
        "base_url": "http://localhost:8001/v1",
        "model": "test-model",
        "timeout": 10,
        "suite": None,
        "score": "logprobs",
        "mode": "closed",
    }
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


def _fake_score_case(case, *, base_url, model, top_logprobs=20, timeout=60):
    """Canned score_case: deterministic, no network."""
    return {
        "answer": case.answer,
        "echo_available": True,
        "headline": 0.8,
        "first_token_mass": 0.7,
        "soft_score": 0.8,
        "per_candidate": {c: 1.0 / len(case.candidates) for c in case.candidates},
    }


# ---------------------------------------------------------------------------
# AC1 — per-case fields present: soft_score, headline, first_token_mass, answer
# ---------------------------------------------------------------------------


def test_eval_cat_json_per_case_fields(tmp_path, monkeypatch, capsys) -> None:
    """JSON output contains per-case soft_score, headline, first_token_mass, answer."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 1}\n{"seed": 2}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=True)
    rc = eval_cmd.cmd_eval_cat(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "mean_soft_score" in payload
    assert "cases" in payload
    assert len(payload["cases"]) == 2

    for case_result in payload["cases"]:
        assert "soft_score" in case_result
        assert "headline" in case_result
        assert "first_token_mass" in case_result
        assert "answer" in case_result
        assert "seed" in case_result


def test_eval_cat_answer_matches_generated_case(tmp_path, monkeypatch, capsys) -> None:
    """Per-case answer == generate_case(seed=N, mode=...).answer (deterministic)."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 42}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=True, mode="closed")
    eval_cmd.cmd_eval_cat(args)

    payload = json.loads(capsys.readouterr().out)
    case_result = payload["cases"][0]

    # generate_case is deterministic — derive ground truth independently.
    expected_case = generate_case(seed=42, mode="closed")
    assert case_result["answer"] == expected_case.answer


# ---------------------------------------------------------------------------
# AC2 — both modes run end-to-end
# ---------------------------------------------------------------------------


def test_eval_cat_mode_closed(tmp_path, monkeypatch, capsys) -> None:
    """--mode closed runs end-to-end; per-case mode field is 'closed'."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 7}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=True, mode="closed")
    rc = eval_cmd.cmd_eval_cat(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "closed"
    assert payload["cases"][0]["mode"] == "closed"


def test_eval_cat_mode_open(tmp_path, monkeypatch, capsys) -> None:
    """--mode open runs end-to-end; per-case mode field is 'open'."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 7}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=True, mode="open")
    rc = eval_cmd.cmd_eval_cat(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "open"
    assert payload["cases"][0]["mode"] == "open"


def test_eval_cat_both_modes_exit_zero(tmp_path, monkeypatch) -> None:
    """Both open and closed modes return exit code 0 (read-only, no --apply)."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 3}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    for mode in ("open", "closed"):
        args = _make_cat_args(suite=str(suite), json=True, mode=mode)
        rc = eval_cmd.cmd_eval_cat(args)
        assert rc == 0, f"Expected exit 0 for mode={mode!r}"


# ---------------------------------------------------------------------------
# Suite-line mode override
# ---------------------------------------------------------------------------


def test_eval_cat_suite_line_mode_override(tmp_path, monkeypatch) -> None:
    """A 'mode' key in a suite line overrides the --mode CLI flag for that case."""
    suite = tmp_path / "cat.jsonl"
    # First case overrides to 'open'; second uses CLI default 'closed'.
    suite.write_text('{"seed": 5, "mode": "open"}\n{"seed": 6}\n', encoding="utf-8")

    captured_modes: list[str] = []

    def _recording(case, *, base_url, model, top_logprobs=20, timeout=60):
        captured_modes.append(case.mode)
        return _fake_score_case(case, base_url=base_url, model=model, timeout=timeout)

    monkeypatch.setattr(eval_cmd, "score_case", _recording)

    args = _make_cat_args(suite=str(suite), json=True, mode="closed")
    rc = eval_cmd.cmd_eval_cat(args)

    assert rc == 0
    assert captured_modes[0] == "open"  # suite-line override
    assert captured_modes[1] == "closed"  # CLI default


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_eval_cat_missing_suite(tmp_path) -> None:
    """Missing suite file raises ModelGearError with EXIT_USER_ERROR."""
    args = _make_cat_args(suite=str(tmp_path / "does_not_exist.jsonl"))
    with pytest.raises(ModelGearError) as exc:
        eval_cmd.cmd_eval_cat(args)
    assert exc.value.code == EXIT_USER_ERROR
    assert "not found" in exc.value.message.lower()


def test_eval_cat_missing_seed_field(tmp_path) -> None:
    """A suite line missing 'seed' raises ModelGearError with EXIT_USER_ERROR."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"mode": "closed"}\n', encoding="utf-8")

    args = _make_cat_args(suite=str(suite))
    with pytest.raises(ModelGearError) as exc:
        eval_cmd.cmd_eval_cat(args)
    assert exc.value.code == EXIT_USER_ERROR
    assert "seed" in exc.value.message.lower()


def test_eval_cat_malformed_json_line(tmp_path) -> None:
    """A non-JSON suite line raises ModelGearError with EXIT_USER_ERROR."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text("not-json\n", encoding="utf-8")

    args = _make_cat_args(suite=str(suite))
    with pytest.raises(ModelGearError) as exc:
        eval_cmd.cmd_eval_cat(args)
    assert exc.value.code == EXIT_USER_ERROR


# ---------------------------------------------------------------------------
# Text-mode output
# ---------------------------------------------------------------------------


def test_eval_cat_text_mode_output(tmp_path, monkeypatch, capsys) -> None:
    """Text mode emits per-case info and a 'mean soft-score' summary line."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 1}\n{"seed": 2}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=False)
    rc = eval_cmd.cmd_eval_cat(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "mean soft-score" in out.lower()


# ---------------------------------------------------------------------------
# Aggregate: mean_soft_score
# ---------------------------------------------------------------------------


def test_eval_cat_mean_soft_score(tmp_path, monkeypatch, capsys) -> None:
    """JSON report includes the correct mean_soft_score across all cases."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text('{"seed": 1}\n{"seed": 2}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=True)
    eval_cmd.cmd_eval_cat(args)

    payload = json.loads(capsys.readouterr().out)
    # _fake_score_case always returns soft_score=0.8, so mean == 0.8.
    assert abs(payload["mean_soft_score"] - 0.8) < 1e-6


# ---------------------------------------------------------------------------
# Blank / comment lines are skipped
# ---------------------------------------------------------------------------


def test_eval_cat_blank_and_comment_lines(tmp_path, monkeypatch, capsys) -> None:
    """Blank lines and '#' comments are skipped without error."""
    suite = tmp_path / "cat.jsonl"
    suite.write_text(
        '# comment\n\n{"seed": 10}\n\n# another comment\n',
        encoding="utf-8",
    )

    monkeypatch.setattr(eval_cmd, "score_case", _fake_score_case)

    args = _make_cat_args(suite=str(suite), json=True)
    rc = eval_cmd.cmd_eval_cat(args)

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["cases"]) == 1


# ---------------------------------------------------------------------------
# Smoke: cmd_eval_minor is unaffected by adding cat
# ---------------------------------------------------------------------------


def test_eval_minor_smoke(tmp_path, monkeypatch, capsys) -> None:
    """cmd_eval_minor still works after the cat sub-verb was added."""
    suite = tmp_path / "suite.jsonl"
    suite.write_text('{"prompt": "hello", "expect_substring": "world"}\n', encoding="utf-8")

    monkeypatch.setattr(eval_cmd, "chat_text", lambda *a, **kw: "hello, world!")

    args = types.SimpleNamespace(
        json=False,
        base_url="http://localhost:8001/v1",
        model="test-model",
        timeout=10,
        suite=str(suite),
    )
    rc = eval_cmd.cmd_eval_minor(args)
    assert rc == 0


# ---------------------------------------------------------------------------
# register() includes cat sub-command
# ---------------------------------------------------------------------------


def test_register_creates_cat_parser() -> None:
    """register() attaches a 'cat' sub-command under eval with the right handler."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cmd.register(sub)

    args = parser.parse_args(["eval", "cat", "--suite", "my.jsonl"])
    assert args.suite == "my.jsonl"
    assert args.func is eval_cmd.cmd_eval_cat


def test_register_cat_mode_choices(tmp_path) -> None:
    """The --mode argument accepts 'open' and 'closed'."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cmd.register(sub)

    for mode in ("open", "closed"):
        args = parser.parse_args(["eval", "cat", "--suite", "s.jsonl", "--mode", mode])
        assert args.mode == mode


def test_register_cat_score_default() -> None:
    """The default --score is 'logprobs'."""
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    eval_cmd.register(sub)

    args = parser.parse_args(["eval", "cat", "--suite", "s.jsonl"])
    assert args.score == "logprobs"
