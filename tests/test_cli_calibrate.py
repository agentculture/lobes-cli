"""Tests for ``lobes calibrate`` (capacity-relative-pool-routing, issue #199, t9).

Ramps concurrency against a role and reports the measured knee via
:func:`lobes.assess.calibration_knee`. Read-only by default; ``--apply``
writes the measured concurrency to ``.env`` as ``<PREFIX>_MAX_ACTIVE`` — and
refuses to when the ramp never demonstrated a genuine plateau.

Most tests are hermetic: :func:`lobes.cli._commands.calibrate.run_concurrent`
is monkeypatched at its imported name — no HTTP, no docker, no live engine
(mirrors ``tests/test_benchmark_all_lobes.py``'s pattern). The autouse
``offline_runtime`` fixture in ``tests/conftest.py`` already neutralises every
other external probe (docker, ``/health``, the live-``/capabilities`` probe).
One block below (the F4 auth-propagation acceptance test) deliberately does
NOT monkeypatch ``run_concurrent`` — it needs the REAL
:class:`concurrent.futures.ThreadPoolExecutor` fan-out to exercise the bug —
and instead runs a real loopback HTTP server, mirroring
``tests/test_cli_gateway_auth.py``'s pattern.
"""

from __future__ import annotations

import http.server
import json
import threading

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
        # aggregate_tok_s = concurrency * (1000 / mpt); reproduced here via
        # total_completion_tokens / total_s (total_s pinned to 1.0) so the
        # canned plateau/rising shapes stay numerically identical under the
        # real (F3-fixed) `_aggregate_tok_s`, which reads THOSE two fields
        # rather than `ms_per_token`.
        aggregate_tok_s = concurrency * (1000.0 / mpt)
        return {
            "concurrency": concurrency,
            "requests_per_s": round(concurrency / (mpt / 1000.0), 3),
            "p50_latency_ms": mpt,
            "p95_latency_ms": mpt,
            "ms_per_token": mpt,
            "total_s": 1.0,
            "total_completion_tokens": aggregate_tok_s,
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
    captured = capsys.readouterr()
    err = json.loads(captured.err)
    assert "never plateaued" in err["message"] or "never plateaued" in err.get("remediation", "")
    # Nothing was written.
    assert _read_env(tmp_path, "PRIMARY_MAX_ACTIVE") is None
    # Qodo F8 pin: a refused --apply must not write a success-shaped result
    # to stdout — automation that only reads stdout must never see a
    # misleading payload despite the nonzero exit.
    assert captured.out == ""


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
    assert capsys.readouterr().out == ""  # Qodo F8 pin


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
# Qodo review finding F7 (#221) — non-finite (NaN/Infinity) and non-positive
# values must be REJECTED for every new numeric argument this verb added, at
# the CLI layer, before any ramp runs. `float("nan")`/`float("inf")` both
# parse cleanly through argparse's own `type=float` — a NaN `--ttft-bound-s`
# makes every `ttft_s > ttft_bound_s` comparison False, silently disabling
# the guard, so a bad flag must fail loudly here instead.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf", "0", "-1.0"])
def test_calibrate_rejects_non_finite_or_non_positive_ttft_bound(tmp_path, capsys, bad_value):
    rc = main(
        [
            "calibrate",
            "cortex",
            f"--ttft-bound-s={bad_value}",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "ttft-bound-s" in err["message"]


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf"])
def test_calibrate_rejects_non_finite_min_relative_gain(tmp_path, capsys, bad_value):
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            f"--min-relative-gain={bad_value}",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "min-relative-gain" in err["message"]


@pytest.mark.parametrize("bad_value", ["0", "-1"])
def test_calibrate_rejects_non_positive_max_tokens(tmp_path, capsys, bad_value):
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--max-tokens",
            bad_value,
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "max-tokens" in err["message"]


@pytest.mark.parametrize("bad_value", ["nan", "inf", "-inf", "0", "-5"])
def test_calibrate_rejects_non_finite_or_non_positive_timeout(tmp_path, capsys, bad_value):
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            f"--timeout={bad_value}",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc != 0
    err = json.loads(capsys.readouterr().err)
    assert "timeout" in err["message"]


def test_calibrate_accepts_finite_positive_min_relative_gain(tmp_path, monkeypatch):
    """Sanity check: a normal, valid --min-relative-gain is NOT rejected."""
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(
        calibrate_cmd,
        "run_concurrent",
        _make_fake_run_concurrent(ms_per_token=_PLATEAU_MS_PER_TOKEN, ttft_ms=_LOW_TTFT),
    )
    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--min-relative-gain",
            "0.1",
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 0


def test_require_finite_rejects_nan_and_infinity_directly() -> None:
    with pytest.raises(calibrate_cmd.ModelGearError):
        calibrate_cmd._require_finite(float("nan"), "--x")
    with pytest.raises(calibrate_cmd.ModelGearError):
        calibrate_cmd._require_finite(float("inf"), "--x")
    with pytest.raises(calibrate_cmd.ModelGearError):
        calibrate_cmd._require_finite(0.0, "--x")
    assert calibrate_cmd._require_finite(1.5, "--x") == 1.5
    assert calibrate_cmd._require_finite(None, "--x", allow_none=True) is None


# ---------------------------------------------------------------------------
# Unit tests on the pure/driver helpers directly (no CLI dispatch)
# ---------------------------------------------------------------------------


def test_aggregate_tok_s_from_run_concurrent_row() -> None:
    row = {"total_completion_tokens": 400, "total_s": 10.0}
    assert calibrate_cmd._aggregate_tok_s(row) == pytest.approx(40.0)


def test_aggregate_tok_s_zero_when_degenerate() -> None:
    row = {"total_completion_tokens": 400, "total_s": 0.0}
    assert calibrate_cmd._aggregate_tok_s(row) == 0.0


def test_aggregate_tok_s_uses_total_tokens_over_wall_time_not_mean_reciprocal() -> None:
    """Qodo F3 pin: the aggregate must be total_completion_tokens / total_s,
    NOT concurrency * (1000 / ms_per_token) — the two diverge sharply
    whenever completion lengths/latencies are unequal across the batch."""
    row = {
        "concurrency": 2,
        "ms_per_token": 10.0,
        "total_completion_tokens": 110,
        "total_s": 1.0,
    }
    assert calibrate_cmd._aggregate_tok_s(row) == pytest.approx(110.0)
    # The old (wrong) formula would have given 2 * (1000 / 10) = 200.0.
    old_wrong_formula = row["concurrency"] * (1000.0 / row["ms_per_token"])
    assert old_wrong_formula == pytest.approx(200.0)
    assert calibrate_cmd._aggregate_tok_s(row) != pytest.approx(old_wrong_formula)


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


# ---------------------------------------------------------------------------
# Qodo review finding F4 (#221) — cmd_calibrate() installs the gateway's
# Authorization header via a ContextVar (lobes.assess.auth_headers), but
# run_concurrent's ThreadPoolExecutor workers do NOT inherit the caller's
# context on their own. This block does NOT monkeypatch run_concurrent — it
# needs the real ThreadPoolExecutor fan-out to exercise the bug — and instead
# runs a real, header-capturing loopback HTTP server end to end through
# `lobes.cli.main`, mirroring `tests/test_cli_gateway_auth.py`'s pattern.
# ---------------------------------------------------------------------------

_CALIBRATE_AUTH_KEY = "calibrate-s3cr3t"


class _CalibrateAuthCapturingHandler(http.server.BaseHTTPRequestHandler):
    """Records every request's Authorization header; always answers 200."""

    seen: list = []

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length:
            self.rfile.read(length)
        type(self).seen.append(self.headers.get("Authorization"))
        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - only /health is hit, pre-dispatch
        if self.path == "/health":
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *_a) -> None:  # silence test noise
        pass


@pytest.fixture
def calibrate_auth_server():
    handler = type(
        "_BoundCalibrateAuthCapturingHandler", (_CalibrateAuthCapturingHandler,), {"seen": []}
    )
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1], handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_calibrate_propagates_gateway_auth_header_to_every_ramp_request(
    tmp_path, calibrate_auth_server
) -> None:
    """Every outbound request `lobes calibrate` fires — throughput AND TTFT,
    across every concurrent worker — must carry the deployment's configured
    Authorization header, even though `run_concurrent` fans them out across
    `ThreadPoolExecutor` worker threads that do not automatically inherit the
    dispatching thread's `contextvars.Context`."""
    port, handler = calibrate_auth_server
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "VLLM_PORT", str(port))
    _env.set_env(tmp_path / _compose.ENV_FILE, "GATEWAY_API_KEY", _CALIBRATE_AUTH_KEY)

    rc = main(
        [
            "calibrate",
            "cortex",
            "--ttft-bound-s",
            "5.0",
            "--schedule",
            "3",  # one level, but concurrency=3 -> 3 workers per call
            "--compose-dir",
            str(tmp_path),
            "--json",
        ]
    )
    assert rc == 0
    # One throughput call + one TTFT call, each fanning out to 3 workers.
    assert len(handler.seen) == 6
    assert all(auth == f"Bearer {_CALIBRATE_AUTH_KEY}" for auth in handler.seen)
