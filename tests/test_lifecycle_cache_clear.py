"""t11: page-cache clear + start-counter reset at the _compose chokepoint."""

from __future__ import annotations

import types

import pytest

from lobes.cli import main
from lobes.runtime import _compose, _machine


@pytest.fixture
def rec(monkeypatch):
    events: list[str] = []
    result = {"ok": True}

    def fake_clear():
        events.append("clear")
        if result["ok"]:
            return _machine.MachineResult(True)
        return _machine.MachineResult(False, "not_granted")

    monkeypatch.setattr(_machine, "drop_caches", fake_clear)

    def fake_run(argv, **kw):
        events.append("docker " + " ".join(argv[2:]))
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(_compose, "_run", fake_run)
    monkeypatch.setattr(_compose, "_cleared_after_down", False)
    # the clear is only attempted on spark/thor/orin (review-b #10)
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: "spark")
    return events, result


def _deploy(tmp_path):
    (tmp_path / "docker-compose.yml").write_text(
        "services:\n  vllm-primary:\n    environment:\n      - MG_LOG_NAME=primary\n"
        "  vllm-hand:\n    environment:\n      - MG_LOG_NAME=hand\n",
        encoding="utf-8",
    )
    state = tmp_path / "logs" / "lane-state"
    state.mkdir(parents=True)
    for n in ("primary", "hand"):
        (state / f"{n}.starts").write_text("1\n", encoding="utf-8")
    return state


def test_down_then_clear(rec, tmp_path):
    events, _ = rec
    _compose.compose_down(tmp_path)
    assert events == ["docker down", "clear"]


def test_clear_before_up(rec, tmp_path):
    events, _ = rec
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]
    events.clear()
    _compose.compose_up_build(tmp_path)
    assert events == ["clear", "docker up -d --build"]


def test_switch_pair_clears_once(rec, tmp_path):
    events, _ = rec
    _compose.compose_down(tmp_path)
    _compose.compose_up_detached(tmp_path)
    assert events == ["docker down", "clear", "docker up -d"]


def test_run_compose_up_and_down(rec, tmp_path):
    events, _ = rec
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert events == ["clear", "docker up -d --no-deps vllm-hand"]
    events.clear()
    _compose.run_compose(tmp_path, ["docker", "compose", "stop", "vllm-hand"])
    assert events == ["docker stop vllm-hand"]  # stop never clears


def test_ungranted_skips_with_one_line_and_continues(rec, tmp_path, capsys):
    events, result = rec
    result["ok"] = False
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]
    err = capsys.readouterr().err
    assert err.count("page-cache clear skipped (not_granted)") == 1
    assert err.count("\n") == 1


def test_targeted_up_resets_only_that_lane(rec, tmp_path):
    state = _deploy(tmp_path)
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert not (state / "hand.starts").exists()
    assert (state / "primary.starts").exists()


def test_whole_project_up_resets_all(rec, tmp_path):
    state = _deploy(tmp_path)
    _compose.compose_up_detached(tmp_path)
    assert list(state.glob("*.starts")) == []


def test_log_dir_from_env_and_missing_ok(rec, tmp_path):
    (tmp_path / ".env").write_text("MODEL_GEAR_LOG_DIR=elsewhere\n", encoding="utf-8")
    d = tmp_path / "elsewhere" / "lane-state"
    d.mkdir(parents=True)
    (d / "x.starts").write_text("1\n", encoding="utf-8")
    _compose.compose_up_detached(tmp_path)
    assert not (d / "x.starts").exists()
    _compose.reset_start_counters(tmp_path / "nonexistent")  # no raise


def test_dry_run_shows_step_and_runs_nothing(rec, tmp_path, capsys):
    events, _ = rec
    _compose.write_scaffold(tmp_path, force=True)
    for argv in (["serve"], ["stop"]):
        assert main([*argv, "--compose-dir", str(tmp_path)]) == 0
        assert "page cache" in capsys.readouterr().out
    assert events == []


# --- review-b #10: card-gated clear + per-process detection cache ----------


@pytest.mark.parametrize("card", ["unknown", "dgx-station", None])
def test_clear_skipped_silently_off_supported_cards(rec, monkeypatch, capsys, card):
    events, _ = rec
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: card)
    probes: list[list[str]] = []
    monkeypatch.setattr(_machine, "_run", lambda argv, **k: probes.append(argv))
    assert _compose.clear_page_cache() is False
    assert events == [] and probes == []  # no drop_caches, no `sudo -n -l`
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("card", ["spark", "thor", "orin"])
def test_clear_attempted_on_supported_cards(rec, monkeypatch, card):
    events, _ = rec
    monkeypatch.setattr(_machine, "detected_card", lambda *a, **k: card)
    assert _compose.clear_page_cache() is True
    assert events == ["clear"]


def test_card_detection_cached_per_process(monkeypatch):
    calls: list[int] = []

    def fake_detect():
        calls.append(1)
        return types.SimpleNamespace(resolved="orin")

    monkeypatch.setattr(_machine, "detect_card", fake_detect)
    assert [_machine.detected_card() for _ in range(5)] == ["orin"] * 5
    _machine._current_cli()  # resolves through the same cache
    assert len(calls) == 1


def test_card_detection_failure_is_none_and_rate_limited(monkeypatch):
    calls: list[int] = []
    now = {"t": 1000.0}
    monkeypatch.setattr(_machine, "_clock", lambda: now["t"])

    def boom():
        calls.append(1)
        raise RuntimeError("nvidia-smi exploded")

    monkeypatch.setattr(_machine, "detect_card", boom)
    assert _machine.detected_card() is None
    assert _machine.detected_card() is None
    assert len(calls) == 1  # inside the retry interval: no re-probe


def _flaky(results):
    calls: list[int] = []

    def detect():
        calls.append(1)
        r = results[min(len(calls), len(results)) - 1]
        if isinstance(r, Exception):
            raise r
        return types.SimpleNamespace(resolved=r)

    return detect, calls


@pytest.mark.parametrize("first", [RuntimeError("nvidia-smi not ready"), "unknown"])
def test_failed_detection_is_retried_after_the_interval(monkeypatch, first):
    """review2-b #6: a boot-time failure must not stick for the watcher's lifetime."""
    now = {"t": 1000.0}
    monkeypatch.setattr(_machine, "_clock", lambda: now["t"])
    detect, calls = _flaky([first, "thor"])
    monkeypatch.setattr(_machine, "detect_card", detect)
    assert _machine.detected_card() in (None, "unknown")
    now["t"] += _machine.DETECT_RETRY_S - 1
    assert _machine.detected_card() in (None, "unknown")
    assert len(calls) == 1
    now["t"] += 2
    assert _machine.detected_card() == "thor"
    now["t"] += 10 * _machine.DETECT_RETRY_S
    assert _machine.detected_card() == "thor"
    assert len(calls) == 2  # a success is cached for good


# --- review-b #3: no lifecycle clear while another lane is mid-boot --------


def _ps(monkeypatch, stdout: str | None):
    calls: list[list[str]] = []

    def probe(argv, **k):
        calls.append(list(argv))
        if stdout is None:
            return None
        return types.SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(_compose, "_probe", probe)
    return calls


def test_up_defers_clear_while_another_lane_starts(rec, tmp_path, monkeypatch, capsys):
    events, _ = rec
    _deploy(tmp_path)
    calls = _ps(
        monkeypatch,
        "vllm-primary\tmodel-gear-vllm-primary\tUp 2 minutes (health: starting)\n"
        "vllm-hand\tmodel-gear-vllm-hand\tUp 3 hours (healthy)\n",
    )
    _compose.run_compose(tmp_path, ["docker", "compose", "up", "-d", "--no-deps", "vllm-hand"])
    assert events == ["docker up -d --no-deps vllm-hand"]  # no clear, verb still runs
    assert calls == [_compose.LANE_HEALTH_ARGV]
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "deferred" in err and "primary" in err


def test_up_clears_when_no_lane_is_starting(rec, tmp_path, monkeypatch):
    events, _ = rec
    _deploy(tmp_path)
    _ps(monkeypatch, "vllm-primary\tmodel-gear-vllm-primary\tUp 2 hours (healthy)\n")
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


def test_gated_lane_starting_does_not_defer_the_clear(rec, tmp_path, monkeypatch):
    events, _ = rec
    state = _deploy(tmp_path)
    (state / "primary.gated").write_text('{"lane": "primary"}', encoding="utf-8")
    _ps(monkeypatch, "vllm-primary\tmodel-gear-vllm-primary\tUp 2 minutes (health: starting)\n")
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


def test_unreadable_docker_still_clears(rec, tmp_path, monkeypatch):
    events, _ = rec
    _ps(monkeypatch, None)
    _compose.compose_up_detached(tmp_path)
    assert events == ["clear", "docker up -d"]


def test_lane_health_maps_services_to_log_names(tmp_path, monkeypatch):
    _deploy(tmp_path)
    _ps(
        monkeypatch,
        "vllm-primary\tmodel-gear-vllm-primary\tUp 1 minute (health: starting)\n"
        "vllm-hand\tmodel-gear-vllm-hand\tUp 2 hours (unhealthy)\n"
        "\tmodel-gear-loose\tUp 5 seconds\n",
    )
    assert _compose.lane_health(tmp_path) == {
        "primary": "starting",
        "hand": "running",
        "model-gear-loose": "running",
    }


def test_lane_health_none_on_docker_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        _compose,
        "_probe",
        lambda argv, **k: types.SimpleNamespace(returncode=1, stdout="", stderr="denied"),
    )
    assert _compose.lane_health(tmp_path) is None
