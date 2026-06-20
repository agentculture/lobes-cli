"""Tests for ``model logs`` + the durable-log helpers (issue #50).

Pure file I/O — no docker. A deployment is scaffolded into a tmp dir (so
``resolve_deployment_dir`` finds a compose file) and fake per-boot log files are
written with controlled mtimes to assert newest-first ordering and crash-boot
recovery via ``--previous``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from model_gear.cli import main
from model_gear.cli._commands import logs as logs_cmd
from model_gear.runtime import _compose

# --- log dir resolution ----------------------------------------------------


def test_durable_log_dir_default(tmp_path) -> None:
    # Unset → <deploy>/logs (matches the compose default ${MODEL_GEAR_LOG_DIR:-./logs}).
    assert _compose.durable_log_dir(tmp_path) == tmp_path / "logs"


def test_durable_log_dir_relative_resolves_against_deploy(tmp_path) -> None:
    assert _compose.durable_log_dir(tmp_path, "mylogs") == tmp_path / "mylogs"


def test_durable_log_dir_absolute_wins(tmp_path) -> None:
    abs_dir = tmp_path / "elsewhere"
    assert _compose.durable_log_dir(tmp_path, str(abs_dir)) == abs_dir


def test_ensure_log_dir_creates(tmp_path) -> None:
    d = _compose.ensure_log_dir(tmp_path)
    assert d.is_dir() and d == tmp_path / "logs"


# --- collect_logs / tail_lines (pure) --------------------------------------


def _boot(log_dir: Path, name: str, mtime: float, body: str = "x") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / name
    p.write_text(body, encoding="utf-8")
    os.utime(p, (mtime, mtime))
    return p


def test_collect_logs_newest_first_and_skips_latest_symlink(tmp_path) -> None:
    d = tmp_path / "logs"
    _boot(d, "vllm-20260620T060000Z.log", 1000.0)
    newest = _boot(d, "vllm-20260620T070000Z.log", 2000.0)
    # a -latest.log pointer must be skipped (it duplicates a real boot)
    _boot(d, "vllm-latest.log", 2000.0)
    got = logs_cmd.collect_logs(d)
    names = [e["name"] for e in got]
    assert names == [newest.name, "vllm-20260620T060000Z.log"]
    assert "vllm-latest.log" not in names


def test_collect_logs_filters_by_service(tmp_path) -> None:
    d = tmp_path / "logs"
    _boot(d, "vllm-20260620T060000Z.log", 1000.0)
    _boot(d, "embed-20260620T060000Z.log", 1001.0)
    assert [e["service"] for e in logs_cmd.collect_logs(d, "embed")] == ["embed"]


def test_collect_logs_missing_dir_is_empty(tmp_path) -> None:
    assert logs_cmd.collect_logs(tmp_path / "nope") == []


def test_tail_lines_returns_last_n(tmp_path) -> None:
    p = tmp_path / "a.log"
    p.write_text("\n".join(f"line{i}" for i in range(100)), encoding="utf-8")
    assert logs_cmd.tail_lines(p, 3) == "line97\nline98\nline99"


def test_tail_lines_window_on_large_file(tmp_path) -> None:
    p = tmp_path / "big.log"
    p.write_text("HEAD\n" + ("z" * 500000) + "\nTAILMARK", encoding="utf-8")
    out = logs_cmd.tail_lines(p, 1, max_bytes=1024)
    assert out == "TAILMARK"  # only the final window is read


# --- the CLI verb ----------------------------------------------------------


def _deploy_with_logs(tmp_path, capsys) -> Path:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    d = target / "logs"
    _boot(d, "vllm-20260620T060000Z.log", 1000.0, body="OLD crash\nTraceback: EngineCore boom")
    _boot(d, "vllm-20260620T070000Z.log", 2000.0, body="NEW healthy boot")
    capsys.readouterr()  # drop init's stdout so the asserted command's output is clean
    return target


def test_logs_list_text(tmp_path, capsys) -> None:
    target = _deploy_with_logs(tmp_path, capsys)
    assert main(["logs", "--compose-dir", str(target)]) == 0
    out = capsys.readouterr().out
    assert "vllm-20260620T070000Z.log" in out
    assert "<- latest" in out  # newest boot flagged


def test_logs_list_json(tmp_path, capsys) -> None:
    target = _deploy_with_logs(tmp_path, capsys)
    assert main(["logs", "--compose-dir", str(target), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    # newest first
    assert payload["files"][0]["name"] == "vllm-20260620T070000Z.log"


def test_logs_tail_latest(tmp_path, capsys) -> None:
    target = _deploy_with_logs(tmp_path, capsys)
    assert main(["logs", "vllm", "--compose-dir", str(target)]) == 0
    assert "NEW healthy boot" in capsys.readouterr().out


def test_logs_previous_shows_crash_boot(tmp_path, capsys) -> None:
    # The whole point of #50: after a restart, --previous tails the boot that crashed.
    target = _deploy_with_logs(tmp_path, capsys)
    assert main(["logs", "vllm", "--previous", "--compose-dir", str(target)]) == 0
    out = capsys.readouterr().out
    assert "EngineCore boom" in out


def test_logs_previous_single_boot_notes(tmp_path, capsys) -> None:
    # --previous with no earlier boot shows the latest, but says so (review feedback).
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    _boot(target / "logs", "vllm-20260620T060000Z.log", 1000.0, body="only boot")
    capsys.readouterr()
    assert main(["logs", "vllm", "--previous", "--compose-dir", str(target)]) == 0
    assert "only 1 boot" in capsys.readouterr().out


def test_logs_empty_dir_friendly(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    assert main(["logs", "--compose-dir", str(target)]) == 0
    assert "no durable logs yet" in capsys.readouterr().out


def test_logs_unknown_service_no_crash(tmp_path, capsys) -> None:
    target = _deploy_with_logs(tmp_path, capsys)
    assert main(["logs", "nosuchsvc", "--compose-dir", str(target)]) == 0
    assert "no logs for 'nosuchsvc'" in capsys.readouterr().out


# --- the shipped wrapper ---------------------------------------------------


def test_logwrap_template_shipped_and_safe() -> None:
    from importlib.resources import files

    content = (files("model_gear.templates") / _compose.LOG_WRAPPER).read_text()
    # Always execs the real command (so logging can never block serving) ...
    assert 'exec "$@"' in content
    # ... tees to a durable file, and is parameterised per service.
    assert "tee -a" in content
    assert "MG_LOG_NAME" in content and "MG_LOG_DIR" in content
    # In both template sets (single + fleet).
    assert "mg-logwrap.sh" in _compose.SINGLE_TEMPLATES
    assert "mg-logwrap.sh" in _compose.FLEET_TEMPLATES


def test_compose_log_dir_does_not_drift_from_python() -> None:
    # Guard: the compose host-dir default and the in-container path must stay in sync
    # with the Python helpers that read them (review feedback).
    from importlib.resources import files

    compose = (files("model_gear.templates") / "docker-compose.yml").read_text()
    assert f"${{MODEL_GEAR_LOG_DIR:-./{_compose.LOG_DIRNAME}}}:/logs/model-gear" in compose
    assert "MG_LOG_DIR=/logs/model-gear" in compose
