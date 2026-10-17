"""Benchmark window: lock/restore ordering, host lock, signals, no-CLI."""

from __future__ import annotations

import os
import signal
from pathlib import Path

import pytest

from lobes.cli._errors import ModelGearError
from lobes.runtime import _machine, _window
from lobes.runtime._machine import MachineResult

ROOT = Path(__file__).resolve().parent.parent
SHOW = {"power_mode": "MAXN", "clocks": {"gpu_mhz": 1}, "fan_profile": "cool", "temps_c": {}}


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    calls: list[str] = []
    results: dict[str, MachineResult] = {}

    def window(action):
        calls.append(action)
        return results.get(action, MachineResult(True, None, dict(SHOW), "machine-cli"))

    monkeypatch.setattr(_machine, "window", window)
    return calls, results


def test_success_order(fake):
    calls, _ = fake
    with _window.benchmark_window() as cond:
        calls.append("run")
    assert calls == ["show", "store", "lock", "run", "restore", "show"]
    d = cond.as_dict()
    assert d["locked"] is True and d["reason"] is None
    assert d["start"]["power_mode"] == "MAXN" and d["end"]["fan_profile"] == "cool"


def test_exception_restores_once(fake):
    calls, _ = fake
    with pytest.raises(RuntimeError):
        with _window.benchmark_window():
            raise RuntimeError("boom")
    assert calls.count("restore") == 1 and calls[-1] == "show"


def test_sigint_restores_once(fake):
    calls, _ = fake
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        with _window.benchmark_window():
            os.kill(os.getpid(), signal.SIGINT)
    assert calls.count("restore") == 1
    assert signal.getsignal(signal.SIGINT) is before


def test_sigterm_restores_once(fake):
    calls, _ = fake
    with pytest.raises(SystemExit):
        with _window.benchmark_window():
            os.kill(os.getpid(), signal.SIGTERM)
    assert calls.count("restore") == 1


def test_concurrent_window_refused(fake):
    calls, _ = fake
    with _window.benchmark_window():
        with pytest.raises(ModelGearError) as e:
            with _window.benchmark_window():
                pass
        assert "already active" in e.value.message
        assert calls.count("restore") == 0  # the refused one never restores
    assert calls.count("restore") == 1
    with _window.benchmark_window():  # lock released afterwards
        pass


def test_no_cli_runs_unlocked(fake):
    calls, results = fake
    results["show"] = MachineResult(False, "tool_missing")
    results["store"] = MachineResult(False, "tool_missing")
    with _window.benchmark_window() as cond:
        pass
    assert calls == ["show", "store", "show"]
    d = cond.as_dict()
    assert d["locked"] is False and d["reason"] == "tool_missing"
    assert d["start"] == {"available": False, "reason": "tool_missing"}


def test_lock_failure_still_restores(fake):
    calls, results = fake
    results["lock"] = MachineResult(False, "not_granted")
    with _window.benchmark_window() as cond:
        pass
    assert calls.count("restore") == 1
    assert cond.as_dict()["reason"] == "not_granted"


def test_no_lock_flag(fake):
    calls, _ = fake
    with _window.benchmark_window(lock_clocks=False) as cond:
        pass
    assert calls == ["show", "show"] and cond.as_dict()["reason"] == "disabled"


def test_benchmark_json_has_conditions(fake, monkeypatch, capsys):
    import json

    import lobes.cli._commands.benchmark as b
    from lobes.cli import main

    monkeypatch.setattr(b._assess, "run_benchmark", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(b._compose, "container_image", lambda: "img")
    monkeypatch.setattr(b._compose, "gpu_engine_mem", lambda: "1G")
    main(["benchmark", "--json", "--model", "m", "--port", "1"])
    out = json.loads(capsys.readouterr().out)
    assert out["host"]["image"] == "img"
    assert out["host"]["conditions"]["locked"] is True


def test_other_verbs_never_use_window():
    files = [ROOT / "lobes/runtime/_compose.py"] + [
        ROOT / f"lobes/cli/_commands/{n}.py"
        for n in ("init", "serve", "switch", "up", "fleet", "stop")
    ]
    for f in files:
        if not f.exists():
            continue
        text = f.read_text()
        assert "_window" not in text and "window(" not in text, f
