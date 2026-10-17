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
    assert d["locked"] is True
    assert d["reason"] is None
    assert d["start"]["power_mode"] == "MAXN"
    assert d["end"]["fan_profile"] == "cool"


def test_exception_restores_once(fake):
    calls, _ = fake
    with pytest.raises(RuntimeError):
        with _window.benchmark_window():
            raise RuntimeError("boom")
    assert calls.count("restore") == 1
    assert calls[-1] == "show"


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
    assert d["locked"] is False
    assert d["reason"] == "tool_missing"
    assert d["start"] == {"available": False, "reason": "tool_missing"}


def test_lock_failure_still_restores(fake):
    calls, results = fake
    results["lock"] = MachineResult(False, "not_granted")
    with _window.benchmark_window() as cond:
        pass
    assert calls.count("restore") == 1
    assert cond.as_dict()["reason"] == "not_granted"


def test_no_lock_records_conditions_without_a_window(fake):
    """d8: an unlocked run touches no clocks and takes no host lock."""
    calls, _ = fake
    with _window.benchmark_window(lock_clocks=False) as cond:
        with _window.benchmark_window():  # the host lock is free: no window held
            pass
    assert calls == ["show", "show", "store", "lock", "restore", "show", "show"]
    d = cond.as_dict()
    assert d["locked"] is False
    assert d["reason"] == "not_requested"
    assert d["start"]["power_mode"] == "MAXN"
    assert d["end"]["power_mode"] == "MAXN"


def _bench(monkeypatch):
    import lobes.cli._commands.benchmark as b

    monkeypatch.setattr(b._assess, "run_benchmark", lambda *a, **k: {"x": 1})
    monkeypatch.setattr(b._compose, "container_image", lambda: "img")
    monkeypatch.setattr(b._compose, "gpu_engine_mem", lambda: "1G")


def test_benchmark_default_is_read_only_no_window(fake, monkeypatch, capsys):
    """d8: `lobes benchmark` is a read-only verb; locking is opt-in (--lock)."""
    import json

    from lobes.cli import main

    calls, _ = fake
    _bench(monkeypatch)
    main(["benchmark", "--json", "--model", "m", "--port", "1"])
    out = json.loads(capsys.readouterr().out)
    assert out["host"]["image"] == "img"
    assert out["host"]["conditions"]["locked"] is False
    assert out["host"]["conditions"]["reason"] == "not_requested"
    assert calls == ["show", "show"]  # no store / lock / restore


def test_benchmark_lock_flag_locks_and_restores(fake, monkeypatch, capsys):
    import json

    from lobes.cli import main

    calls, _ = fake
    _bench(monkeypatch)
    main(["benchmark", "--lock", "--json", "--model", "m", "--port", "1"])
    out = json.loads(capsys.readouterr().out)
    assert out["host"]["conditions"]["locked"] is True
    assert calls == ["show", "store", "lock", "restore", "show"]


def test_benchmark_no_lock_flag_is_gone(fake, monkeypatch, capsys):
    from lobes.cli import main

    with pytest.raises(SystemExit) as e:
        main(["benchmark", "--no-lock", "--model", "m", "--port", "1"])
    assert e.value.code != 0
    assert "--no-lock" in capsys.readouterr().err


def test_other_verbs_never_use_window():
    files = [ROOT / "lobes/runtime/_compose.py"] + [
        ROOT / f"lobes/cli/_commands/{n}.py"
        for n in ("init", "serve", "switch", "up", "fleet", "stop")
    ]
    for f in files:
        if not f.exists():
            continue
        text = f.read_text()
        assert "_window" not in text, f
        assert "window(" not in text, f
