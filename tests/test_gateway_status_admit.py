"""Tests for the admission signal on the gateway's ``GET /status`` (issue #346).

``/status`` used to say how busy the box is (``busy``, ``pressure``) but not
whether it wants new work, and it carried no timestamp. A caller deciding
whether to start work had to rebuild the rule and could not tell a fresh
answer from a cached one. These tests pin the four additive top-level fields:
``admit``, ``at``, ``stale_after_s`` and ``reason``.

Pure where possible (the probe and the clock are injected); one test drives a
real handler on an ephemeral loopback port to prove the field reaches the wire.
"""

from __future__ import annotations

import json
import re
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lobes.gateway import _admission as A
from lobes.gateway import _pressure_policy as P
from lobes.gateway import server as S
from lobes.gateway._config import ServerConfig
from lobes.gateway._routing import Backend, RoutingTable
from lobes.gateway._tier_request import PressureCache

# 2026-11-26T12:00:00Z as epoch seconds, with a fraction that must be dropped.
_EPOCH = 1795694400.75
_AT = "2026-11-26T12:00:00Z"
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}
_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}
_HIGH_IOWAIT = {"swap_used_percent": 0.0, "iowait_percent": 90.0}
_HOT = {"swap_used_percent": 0.0, "iowait_percent": 0.0, "thermal_c": 99.0}


def _cfg() -> ServerConfig:
    return ServerConfig(host="127.0.0.1", port=8000, connect_timeout=1.0, read_timeout=1.0)


def _table(base_url: str = "http://p:8000") -> RoutingTable:
    return RoutingTable(
        backends=(Backend("primary", base_url, "P", "generate"),),
        default_model="P",
        aliases={},
    )


def _probe(waiting: int = 0, running: int = 0):
    def probe(base_url, *, timeout):  # noqa: ARG001 - signature matches _metrics.probe_backend
        return {"health": "ok", "metrics": {"running": running, "waiting": waiting}}

    return probe


def _status(pressure, *, waiting: int = 0, running: int = 0) -> dict:
    return S.fleet_status_payload(
        _table(), _cfg(), pressure, probe=_probe(waiting, running), clock=lambda: _EPOCH
    )


# --- the pure rule ----------------------------------------------------------


def test_admission_admits_when_nothing_is_shed_and_nothing_waits() -> None:
    out = A.admission(shed=False, shed_signal="", waiting=0, now=_EPOCH)
    assert out == {"admit": True, "at": _AT, "stale_after_s": A.STALE_AFTER_S}
    assert "reason" not in out  # reason is only present on a hold


@pytest.mark.parametrize("signal", ["swap", "thermal", "engine", "iowait"])
def test_admission_holds_when_the_pressure_policy_sheds(signal: str) -> None:
    out = A.admission(shed=True, shed_signal=signal, waiting=0, now=_EPOCH)
    assert out["admit"] is False
    assert out["reason"] == f"shed:{signal}"


def test_admission_names_a_shed_without_a_signal() -> None:
    out = A.admission(shed=True, shed_signal="", waiting=0, now=_EPOCH)
    assert out["admit"] is False
    assert out["reason"] == "shed"


def test_admission_holds_when_requests_are_waiting() -> None:
    out = A.admission(shed=False, shed_signal="", waiting=3, now=_EPOCH)
    assert out["admit"] is False
    assert out["reason"] == "waiting"


def test_admission_reports_the_shed_before_the_queue() -> None:
    # A shedding box with a queue names the shed: it is the more fundamental fact.
    out = A.admission(shed=True, shed_signal="swap", waiting=5, now=_EPOCH)
    assert out["reason"] == "shed:swap"


def test_admission_timestamp_is_rfc3339_utc_to_the_second() -> None:
    out = A.admission(shed=False, shed_signal="", waiting=0, now=_EPOCH)
    assert out["at"] == _AT
    assert _RFC3339_UTC.match(out["at"])
    assert A.admission(shed=False, shed_signal="", waiting=0, now=0.0)["at"] == (
        "1970-01-01T00:00:00Z"
    )


def test_stale_after_s_is_a_positive_float() -> None:
    assert isinstance(A.STALE_AFTER_S, float)
    assert A.STALE_AFTER_S > 0


# --- on the /status payload -------------------------------------------------


def test_status_admits_on_an_idle_box() -> None:
    payload = _status(_NO_PRESSURE)
    assert payload["admit"] is True
    assert payload["at"] == _AT
    assert payload["stale_after_s"] == A.STALE_AFTER_S
    assert "reason" not in payload


def test_status_admits_while_requests_run_but_none_wait() -> None:
    payload = _status(_NO_PRESSURE, running=4)
    assert payload["admit"] is True


def test_status_holds_under_swap_pressure() -> None:
    payload = _status(_HIGH_SWAP)
    assert payload["pressure"]["shed"] is True
    assert payload["admit"] is False
    assert payload["reason"] == "shed:swap"


def test_status_holds_under_iowait_pressure() -> None:
    payload = _status(_HIGH_IOWAIT)
    assert payload["admit"] is False
    assert payload["reason"] == "shed:iowait"


def test_status_holds_when_the_box_is_hot(monkeypatch: pytest.MonkeyPatch) -> None:
    # The thermal band is off unless the deployment declares one.
    monkeypatch.setattr(P, "THERMAL_SHED_C", 91.0)
    monkeypatch.setattr(P, "THERMAL_RESUME_C", 86.0)
    payload = _status(_HOT)
    assert payload["pressure"]["thermal_shedding"] is True
    assert payload["admit"] is False
    assert payload["reason"] == "shed:thermal"


def test_status_holds_when_a_backend_has_a_waiting_queue() -> None:
    payload = _status(_NO_PRESSURE, waiting=2)
    assert payload["busy"]["waiting"] == 2
    assert payload["admit"] is False
    assert payload["reason"] == "waiting"


def test_status_admit_agrees_with_the_pressure_block() -> None:
    # The gateway owns the rule: admit can never be true while it sheds.
    for pressure in (_NO_PRESSURE, _HIGH_SWAP, _HIGH_IOWAIT):
        payload = _status(pressure)
        assert payload["admit"] is (not payload["pressure"]["shed"])


def test_status_omits_the_signal_when_pressure_is_not_wired() -> None:
    # No pressure sample means the gateway cannot answer; a reader that finds no
    # `admit` treats the signal as not available. The legacy payload is unchanged.
    payload = S.fleet_status_payload(_table(), _cfg(), probe=_probe(), clock=lambda: _EPOCH)
    for key in ("admit", "at", "stale_after_s", "reason"):
        assert key not in payload


def test_status_keeps_every_existing_field() -> None:
    payload = _status(_HIGH_SWAP)
    assert payload["object"] == "lobes.fleet_status"
    assert payload["default_model"] == "P"
    assert payload["busy"] == {"running": 0, "waiting": 0}
    assert payload["pressure"]["reason"] == "pressure"  # the nested reason is untouched
    assert [b["name"] for b in payload["backends"]] == ["primary"]
    assert "GET /status" in payload["endpoints"]


def test_status_uses_the_wall_clock_by_default() -> None:
    payload = S.fleet_status_payload(_table(), _cfg(), _NO_PRESSURE, probe=_probe())
    assert _RFC3339_UTC.match(payload["at"])


def test_status_payload_is_json_serialisable() -> None:
    again = json.loads(json.dumps(_status(_HIGH_SWAP)))
    assert again["admit"] is False
    assert again["stale_after_s"] == A.STALE_AFTER_S


# --- on the wire ------------------------------------------------------------


def _get_status(pressure: dict) -> dict:
    # Port 1 on loopback refuses at once, so the probe reports "unreachable"
    # without waiting and no real backend is ever contacted.
    cache = PressureCache(sampler=lambda: pressure, start=False)
    handler = S._make_handler(_table("http://127.0.0.1:1"), _cfg(), pressure_cache=cache)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/status", timeout=10) as resp:
            assert resp.status == 200
            return json.loads(resp.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_get_status_carries_the_admission_signal() -> None:
    # The one backend refuses the probe, so the gateway holds: nothing here
    # could serve the work. The pressure sample itself is fresh and calm.
    payload = _get_status(_NO_PRESSURE)
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_UNREACHABLE
    assert _RFC3339_UTC.match(payload["at"])
    assert payload["stale_after_s"] == A.STALE_AFTER_S


def test_get_status_holds_under_pressure() -> None:
    payload = _get_status(_HIGH_SWAP)
    assert payload["admit"] is False
    assert payload["reason"] == "shed:swap"


# --- findings from the review of PR #347 ------------------------------------


def _unreachable_probe(base_url, *, timeout):  # noqa: ARG001 - matches _metrics.probe_backend
    return {"health": "unreachable", "metrics": None}


def test_cache_records_when_it_last_sampled_successfully() -> None:
    cache = PressureCache(sampler=lambda: dict(_NO_PRESSURE), start=False, clock=lambda: _EPOCH)
    assert cache.sampled_at() == _EPOCH


def test_cache_that_never_sampled_has_no_sample_time() -> None:
    def boom() -> dict:
        raise OSError("unreadable")

    assert PressureCache(sampler=boom, start=False, clock=lambda: _EPOCH).sampled_at() is None


def test_status_holds_when_the_pressure_sample_never_succeeded() -> None:
    payload = S.fleet_status_payload(
        _table(), _cfg(), dict(_NO_PRESSURE), probe=_probe(), clock=lambda: _EPOCH, sampled_at=None
    )
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_NO_SAMPLE
    assert payload["at"] == _AT


def test_status_stamps_the_sample_time_not_the_response_time() -> None:
    payload = S.fleet_status_payload(
        _table(),
        _cfg(),
        dict(_NO_PRESSURE),
        probe=_probe(),
        clock=lambda: _EPOCH + 4,
        sampled_at=_EPOCH,
    )
    assert payload["admit"] is True
    assert payload["at"] == _AT


def test_status_holds_when_the_pressure_sample_is_stale() -> None:
    late = _EPOCH + A.STALE_AFTER_S + 1
    payload = S.fleet_status_payload(
        _table(), _cfg(), dict(_NO_PRESSURE), probe=_probe(), clock=lambda: late, sampled_at=_EPOCH
    )
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_STALE
    assert payload["at"] == _AT


def test_status_holds_when_no_backend_is_healthy() -> None:
    payload = S.fleet_status_payload(
        _table(), _cfg(), dict(_NO_PRESSURE), probe=_unreachable_probe, clock=lambda: _EPOCH
    )
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_UNREACHABLE


def test_status_admits_when_one_of_two_backends_is_healthy() -> None:
    table = RoutingTable(
        backends=(
            Backend("primary", "http://p:8000", "P", "generate"),
            Backend("second", "http://q:8000", "Q", "generate"),
        ),
        default_model="P",
        aliases={},
    )

    def probe(base_url, *, timeout):  # noqa: ARG001
        if "//p:" in base_url:
            return {"health": "unreachable", "metrics": None}
        return {"health": "ok", "metrics": {"running": 0, "waiting": 0}}

    payload = S.fleet_status_payload(
        table, _cfg(), dict(_NO_PRESSURE), probe=probe, clock=lambda: _EPOCH
    )
    assert payload["admit"] is True


def test_a_shed_is_reported_before_an_unreachable_fleet() -> None:
    payload = S.fleet_status_payload(
        _table(), _cfg(), dict(_HIGH_SWAP), probe=_unreachable_probe, clock=lambda: _EPOCH
    )
    assert payload["reason"] == "shed:swap"


def test_the_documented_reasons_are_the_ones_status_can_emit() -> None:
    """`shed:engine` is not reachable from /status: it passes no engine load."""
    doc = (
        __import__("pathlib").Path(__file__).parent.parent / "docs" / "gateway-fleet.md"
    ).read_text()
    row = next(line for line in doc.splitlines() if line.lstrip().startswith("| `reason` |"))
    assert "shed:engine" not in row
    for reason in (
        "shed:swap",
        "shed:thermal",
        "shed:iowait",
        "waiting",
        "unreachable",
        "stale",
        "no-sample",
    ):
        assert reason in row


# --- findings from the second review (an OpenCode run through `agents`) ------


def _status_at(now: float, sampled_at, pressure=None, **kw) -> dict:
    return S.fleet_status_payload(
        _table(),
        _cfg(),
        dict(pressure or _NO_PRESSURE),
        probe=_probe(),
        clock=lambda: now,
        sampled_at=sampled_at,
        **kw,
    )


def test_a_sampler_that_fails_after_a_success_reads_as_no_sample_at_once() -> None:
    """The zeros a failed read leaves behind are not a measurement, even 1 s later."""
    answers = [dict(_HIGH_SWAP)]

    def sampler() -> dict:
        if answers:
            return answers.pop()
        raise OSError("unreadable")

    cache = PressureCache(sampler=sampler, start=False, clock=lambda: _EPOCH)
    assert cache.sampled_at() == _EPOCH
    cache.refresh()
    assert cache.current()["swap_used_percent"] == 0.0
    assert cache.sampled_at() is None


def test_a_raising_clock_does_not_escape_the_cache() -> None:
    def clock() -> float:
        raise RuntimeError("clock")

    cache = PressureCache(sampler=lambda: dict(_NO_PRESSURE), start=False, clock=clock)
    assert cache.sampled_at() is None
    cache.refresh()  # must not raise: a raise here would end the refresh thread


def test_a_sample_stamped_in_the_future_is_not_trusted() -> None:
    """After the clock steps back, the stamp says nothing about the sample's age."""
    payload = _status_at(_EPOCH - 60, _EPOCH)
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_STALE


def test_a_stale_sample_is_reported_before_a_shed() -> None:
    payload = _status_at(_EPOCH + A.STALE_AFTER_S + 1, _EPOCH, _HIGH_SWAP)
    assert payload["reason"] == A.REASON_STALE


def test_a_sample_exactly_stale_after_s_old_is_still_trusted() -> None:
    assert _status_at(_EPOCH + A.STALE_AFTER_S, _EPOCH)["admit"] is True


def test_a_sample_time_that_is_not_a_number_holds() -> None:
    payload = _status_at(_EPOCH, "now")
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_NO_SAMPLE


def test_a_fleet_with_no_backends_holds() -> None:
    table = RoutingTable(backends=(), default_model="P", aliases={})
    payload = S.fleet_status_payload(
        table, _cfg(), dict(_NO_PRESSURE), probe=_probe(), clock=lambda: _EPOCH
    )
    assert payload["admit"] is False
    assert payload["reason"] == A.REASON_UNREACHABLE


def test_a_backend_without_in_flight_counts_cannot_fire_waiting() -> None:
    """The documented limit: `busy.partial` admits, because the queue is not measured."""

    def probe(base_url, *, timeout):  # noqa: ARG001
        return {
            "health": "ok",
            "metrics": {"engine": "llamacpp", "unsupported": ["running", "waiting"]},
        }

    payload = S.fleet_status_payload(
        _table(), _cfg(), dict(_NO_PRESSURE), probe=probe, clock=lambda: _EPOCH
    )
    assert payload["busy"].get("partial") is True
    assert payload["admit"] is True


def test_a_pooled_box_is_not_held_on_iowait_alone() -> None:
    """A live pooled request is served under iowait-only pressure, so /status must admit."""
    assert _status_at(_EPOCH, _EPOCH, _HIGH_IOWAIT)["reason"] == "shed:iowait"
    pooled = _status_at(_EPOCH, _EPOCH, _HIGH_IOWAIT, pooled=True)
    assert pooled["admit"] is True
    assert pooled["pressure"]["shed"] is False


def test_a_pooled_box_still_holds_under_swap() -> None:
    assert _status_at(_EPOCH, _EPOCH, _HIGH_SWAP, pooled=True)["reason"] == "shed:swap"


def _wire_status(cache) -> dict:
    handler = S._make_handler(_table("http://127.0.0.1:1"), _cfg(), pressure_cache=cache)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/status", timeout=10) as resp:
            return json.loads(resp.read())
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_on_the_wire_at_is_the_sample_time() -> None:
    cache = PressureCache(sampler=lambda: dict(_NO_PRESSURE), start=False, clock=lambda: _EPOCH)
    payload = _wire_status(cache)
    assert payload["at"] == _AT  # the sample's stamp, not the wall clock of this test
    assert payload["reason"] == A.REASON_STALE  # and that sample is long stale by now


def test_on_the_wire_a_cache_without_a_sample_time_stamps_the_response() -> None:
    class Bare:
        def current(self) -> dict:
            return dict(_NO_PRESSURE)

    payload = _wire_status(Bare())
    assert _RFC3339_UTC.match(payload["at"])
    assert payload["at"] != _AT
    assert payload["reason"] == A.REASON_UNREACHABLE  # the one backend refuses the probe


def test_the_docs_say_what_at_is_when_the_sample_time_is_unknown() -> None:
    doc = (
        __import__("pathlib").Path(__file__).parent.parent / "docs" / "gateway-fleet.md"
    ).read_text()
    row = next(line for line in doc.splitlines() if line.lstrip().startswith("| `at` |"))
    assert "when the status was computed" in row
