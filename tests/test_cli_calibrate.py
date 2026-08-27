"""Tests for ``lobes calibrate`` (capacity-relative-pool-routing, issue #199, t9).

Ramps concurrency against a role and reports the measured knee via
:func:`lobes.assess.calibration_knee`. Read-only by default; ``--apply``
writes the measured concurrency to ``.env`` as ``<PREFIX>_MAX_ACTIVE`` — and
refuses to when the ramp never demonstrated a genuine plateau.

All tests are hermetic: :func:`lobes.cli._commands.calibrate.run_concurrent`
is monkeypatched at its imported name — no HTTP, no docker, no live engine
(mirrors ``tests/test_benchmark_all_lobes.py``'s pattern). The autouse
``offline_runtime`` fixture in ``tests/conftest.py`` already neutralises every
other external probe (docker, ``/health``, the live-``/capabilities`` probe).
"""

from __future__ import annotations

import json

import pytest

import lobes.cli._commands.calibrate as calibrate_cmd
from lobes.cli import main
from lobes.gateway._config import MAX_ACTIVE_ENV
from lobes.runtime import _compose, _env


def _scaffold_fleet(path):
    """The packaged fleet templates verbatim — cortex/senses/embedder/reranker
    all ``loaded`` (mirrors ``tests/test_cli_capabilities.py``'s helper)."""
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def _read_env(path, key):
    return _env.read_env(path / _compose.ENV_FILE, key)


# ---------------------------------------------------------------------------
# Canned run_concurrent fakes — keyed by (concurrency, max_tokens==1 or not)
# ---------------------------------------------------------------------------


def _make_fake_run_concurrent(*, ms_per_token: dict, ttft_ms: dict, calls: list | None = None):
    """Build a fake matching :func:`lobes.assess.run_concurrent`'s call shape.

    ``ms_per_token[concurrency]`` drives the "throughput" call (``max_tokens
    != 1``); ``ttft_ms[concurrency]`` drives the "TTFT" call (``max_tokens ==
    1``) via its ``p50_latency_ms``. ``calls`` (if given) records every
    invocation's ``(concurrency, max_tokens)`` so a test can assert on the
    early-stop behaviour of :func:`~lobes.cli._commands.calibrate.drive_calibration`.
    """

    def _fake(url, model, *, concurrency, max_tokens=128, timeout=300):
        if calls is not None:
            calls.append((concurrency, max_tokens))
        if max_tokens == 1:
            return {
                "concurrency": concurrency,
                "requests_per_s": 1.0,
                "p50_latency_ms": ttft_ms.get(concurrency, 50.0),
                "p95_latency_ms": ttft_ms.get(concurrency, 50.0),
                "ms_per_token": ttft_ms.get(concurrency, 50.0),
                "total_s": 0.1,
            }
        mpt = ms_per_token[concurrency]
        return {
            "concurrency": concurrency,
            "requests_per_s": round(concurrency / (mpt / 1000.0), 3),
            "p50_latency_ms": mpt,
            "p95_latency_ms": mpt,
            "ms_per_token": mpt,
            "total_s": 0.5,
        }

    return _fake


# ms_per_token chosen so aggregate_tok_s = concurrency * 1000 / ms_per_token
# reproduces the plateau shape of tests/test_calibration_knee.py's own
# test_knee_at_throughput_plateau: 10 -> 19 -> 21 -> 21.5 tok/s.
_PLATEAU_MS_PER_TOKEN = {
    1: 100.0,  # agg 10.0
    2: 2000.0 / 19.0,  # agg 19.0
    4: 4000.0 / 21.0,  # agg 21.0
    8: 8000.0 / 21.5,  # agg 21.5 -> plateau (gain ~0.024 < 0.1)
    16: 8000.0 / 21.5,  # never reached if early-stop works
    32: 8000.0 / 21.5,
}
_LOW_TTFT = {c: 50.0 for c in (1, 2, 4, 8, 16, 32)}  # 0.05s, well under any bound used below

# Every level keeps rising well past the 10% gain bar -> never plateaus.
_RISING_MS_PER_TOKEN = {
    1: 100.0,  # agg 10.0
    2: 2000.0 / 25.0,  # agg 25.0 (gain 1.5)
    4: 4000.0 / 60.0,  # agg 60.0 (gain 1.4)
}


def _args(role="cortex", compose_dir=None, extra=None):
    argv = ["calibrate", role, "--ttft-bound-s", "5.0", "--schedule", "1,2,4,8,16,32"]
    if compose_dir is not None:
        argv += ["--compose-dir", str(compose_dir)]
    if extra:
        argv += extra
    return argv


# ---------------------------------------------------------------------------
# Acceptance 1 — reports the knee, samples, and whether the ramp plateaued
# ---------------------------------------------------------------------------


def test_calibrate_reports_plateau_knee_and_samples(tmp_path, capsys, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    calls: list = []
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(
            ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT, calls=calls
        ),
    )
    rc = main(_args(compose_dir=tmp_path, extra=["--json"]))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["role"] == "cortex"
    assert payload["backend"] == "primary"
    assert payload["concurrency"] == 4
    assert payload["plateaued"] is True
    assert payload["stopped_by"] == "plateau"
    assert [s[0] for s in payload["samples"]] == [1, 2, 4]
    assert payload["applied"] is False


def test_calibrate_early_stops_the_ramp_once_the_knee_is_known(tmp_path, monkeypatch) -> None:
    """Levels 16/32 must never be measured once the plateau at 8 is seen."""
    _scaffold_fleet(tmp_path)
    calls: list = []
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(
            ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT, calls=calls
        ),
    )
    rc = main(_args(compose_dir=tmp_path))
    assert rc == 0
    measured_concurrencies = sorted({c for c, _mt in calls})
    assert measured_concurrencies == [1, 2, 4, 8]
    assert 16 not in measured_concurrencies
    assert 32 not in measured_concurrencies


def test_calibrate_text_output_uses_plain_language_for_stopped_by(
    tmp_path, capsys, monkeypatch
) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    rc = main(_args(compose_dir=tmp_path))
    assert rc == 0
    out = capsys.readouterr().out
    assert "plateaued" in out.lower()
    assert "top of its schedule" not in out  # only printed for top_of_ramp
    assert "throughput plateaued" in out


# ---------------------------------------------------------------------------
# Acceptance 2 — read-only by default; --apply follows dry-run convention
# ---------------------------------------------------------------------------


def test_calibrate_is_read_only_by_default_env_untouched(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    before = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    rc = main(_args(compose_dir=tmp_path))
    assert rc == 0
    after = (tmp_path / _compose.ENV_FILE).read_text(encoding="utf-8")
    assert before == after
    assert _read_env(tmp_path, MAX_ACTIVE_ENV["primary"]) is None


def test_calibrate_apply_writes_max_active_when_plateaued(tmp_path, capsys, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    rc = main(_args(compose_dir=tmp_path, extra=["--apply", "--json"]))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["env_key"] == "PRIMARY_MAX_ACTIVE"
    assert _read_env(tmp_path, "PRIMARY_MAX_ACTIVE") == "4"


# ---------------------------------------------------------------------------
# Acceptance 3 — refuses to write a never-plateaued (top-of-ramp) result
# ---------------------------------------------------------------------------


def test_calibrate_apply_refuses_when_ramp_never_plateaus(tmp_path, capsys, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_RISING_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--schedule",
            "1,2,4",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--json",
        ]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "never plateaued" in err["message"] or "never plateaued" in err.get("remediation", "")
    # Nothing was written.
    assert _read_env(tmp_path, "PRIMARY_MAX_ACTIVE") is None


def test_calibrate_dry_run_still_reports_top_of_ramp_result(tmp_path, capsys, monkeypatch) -> None:
    """Without --apply, a never-plateaued ramp is still reported (not an error)."""
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_RISING_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--schedule",
            "1,2,4",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped_by"] == "top_of_ramp"
    assert payload["plateaued"] is False
    assert payload["concurrency"] == 4  # top level tried, reported honestly


def test_calibrate_apply_refuses_zero_concurrency_ttft_bound_violation(
    tmp_path, capsys, monkeypatch
) -> None:
    """Even the lowest concurrency violates the TTFT bound -> concurrency=0; refused."""
    _scaffold_fleet(tmp_path)
    ttft = {1: 9000.0, 2: 9000.0, 4: 9000.0}  # 9s, way over any sane bound
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_RISING_MS_PER_TOKEN, ttft_ms=ttft),
    )
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "1.0",
            "--schedule",
            "1,2,4",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--json",
        ]
    )
    assert rc != 0
    assert _read_env(tmp_path, "PRIMARY_MAX_ACTIVE") is None


def test_calibrate_apply_allows_a_ttft_bound_stop_with_positive_concurrency(
    tmp_path, capsys, monkeypatch
) -> None:
    """A genuine ttft_bound stop (not top_of_ramp, concurrency > 0) IS writable."""
    _scaffold_fleet(tmp_path)
    ttft = {1: 50.0, 2: 2000.0, 4: 2000.0}  # level 2 crosses a 1.0s bound
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_RISING_MS_PER_TOKEN, ttft_ms=ttft),
    )
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "1.0",
            "--schedule",
            "1,2,4",
            "--compose-dir",
            str(tmp_path),
            "--apply",
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["stopped_by"] == "ttft_bound"
    assert payload["concurrency"] == 1
    assert payload["applied"] is True
    assert _read_env(tmp_path, "PRIMARY_MAX_ACTIVE") == "1"


# ---------------------------------------------------------------------------
# Acceptance 4 (structural) — the verb lives in lobes/cli/_commands, no
# gateway import; and role wiring / argument validation
# ---------------------------------------------------------------------------


def test_calibrate_errors_on_unloaded_role_message(tmp_path, capsys, monkeypatch) -> None:
    """``stt`` is declared but not loaded on a fleet scaffold without --audio."""
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    rc = main(
        ["calibrate", "stt", "--ttft-bound-s", "5.0", "--compose-dir", str(tmp_path), "--json"]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "not loaded" in err["message"]


def test_calibrate_requires_ttft_bound_flag(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["calibrate", "cortex", "--compose-dir", str(tmp_path)])
    assert exc.value.code != 0


def test_calibrate_rejects_malformed_schedule(tmp_path, capsys) -> None:
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--schedule",
            "1,x,4",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "schedule" in err["message"]


def test_calibrate_rejects_non_positive_schedule_value(tmp_path, capsys) -> None:
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--schedule",
            "1,0,4",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc != 0


# ---------------------------------------------------------------------------
# Unit tests on the pure/driver helpers directly (no CLI dispatch)
# ---------------------------------------------------------------------------


def test_aggregate_tok_s_from_run_concurrent_row() -> None:
    row = {"concurrency": 4, "ms_per_token": 100.0}
    assert calibrate_cmd._aggregate_tok_s(row) == pytest.approx(40.0)


def test_aggregate_tok_s_zero_when_degenerate() -> None:
    row = {"concurrency": 4, "ms_per_token": 0.0}
    assert calibrate_cmd._aggregate_tok_s(row) == 0.0


def test_drive_calibration_pure_over_injected_measure() -> None:
    """drive_calibration never imports/uses a real network call when _measure is given."""
    fake = _make_fake_run_concurrent(ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT)
    knee = calibrate_cmd.drive_calibration(
        "http://unused.invalid",
        "unused-model",
        schedule=(1, 2, 4, 8, 16, 32),
        ttft_bound_s=5.0,
        _measure=fake,
    )
    assert knee.concurrency == 4
    assert knee.plateaued is True
    assert knee.stopped_by == "plateau"


def test_parse_schedule_default_and_custom() -> None:
    assert calibrate_cmd._parse_schedule(None) == calibrate_cmd._DEFAULT_SCHEDULE
    assert calibrate_cmd._parse_schedule("8,1,4") == (1, 4, 8)  # sorted ascending


def test_no_calibration_logic_imported_from_gateway_package() -> None:
    """Non-goal: no gateway import in the calibrate CLI module (it only ever
    produces a number for the gateway to consume, per the spec's non-goals)."""
    import inspect

    src = inspect.getsource(calibrate_cmd)
    # The one deliberate exception: reading the (already-existing) MAX_ACTIVE_ENV
    # name mapping to know which .env key to write — no gateway request-path
    # logic (selection/replicas) is imported.
    assert "lobes.gateway._selection" not in src
    assert "lobes.gateway._replicas" not in src
    assert "lobes.gateway.server" not in src
