"""End-to-end busy-backpressure tests: full handle_post path (resolve -> shed/serve).

Exercises the complete request lifecycle through ``handle_post``, not just the
``decide()`` policy function.  Verifies that under swap/iowait pressure the
gateway sheds full-tier requests with HTTP 429 (no upstream dialed), and that
the busy signal is transient (a retrying client gets a genuine answer once
pressure clears).

See tests/test_gateway_server.py for the shared fixture style (_opener,
_fleet_cfg, _HIGH_SWAP, _NO_PRESSURE).
"""

from __future__ import annotations

import json

from lobes.gateway import server as S
from lobes.gateway._config import build_config

# --- fixtures (mirror test_gateway_server.py style) -------------------------


def _fleet_cfg():
    """A full three-tier generate fleet with identifiable served names."""
    return build_config(
        {
            "PRIMARY_SERVED_NAME": "PRIMARY",
            # The cheap tier resolves to the `hand` backend now.
            "HAND_BASE_URL": "http://vllm-hand:8000",
            "HAND_SERVED_NAME": "HAND",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
            "MULTIMODAL_SERVED_NAME": "MULTIMODAL",
        }
    )


def _opener(behavior):
    """behavior: {backend_name: status_int | Exception}. Records (name, body)."""
    calls = []

    class _FakeUpstream:
        def __init__(self, status, body=b'{"ok":1}'):
            self.status = status
            self.headers = [("Content-Type", "application/json")]
            self._body = body
            self.closed = False

        def read_all(self):
            return self._body

        def read(self, _n):
            data, self._body = self._body, b""
            return data

        def close(self):
            self.closed = True

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, body))
        outcome = behavior[backend.name]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeUpstream(outcome)

    return opener, calls


_HIGH_SWAP = {"swap_used_percent": 80.0, "iowait_percent": 0.0}
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


# --- e2e busy-backpressure tests -------------------------------------------


def test_cortex_request_under_pressure_is_shed_with_429() -> None:
    """model=cortex under HIGH pressure → 429 busy, no upstream dialed."""
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"hand": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"cortex"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 429
    assert resp.upstream is None
    assert calls == []  # no backend was dialed on the shed path
    headers = dict(resp.headers)
    assert headers["Retry-After"] == str(S.BUSY_RETRY_AFTER_SECONDS)
    assert headers["X-Lobes-Tier-Reason"] == "busy"
    body = json.loads(resp.body)
    assert body["error"]["type"] == "server_busy"
    assert body["error"]["code"] == "busy"


def test_senses_request_under_pressure_is_shed_with_429() -> None:
    """model=senses under HIGH pressure → 429 busy, no upstream dialed.

    Busy covers both cortex AND senses (main + multimodal tiers).
    """
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"hand": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"senses"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 429
    assert resp.upstream is None
    assert calls == []
    headers = dict(resp.headers)
    assert headers["Retry-After"] == str(S.BUSY_RETRY_AFTER_SECONDS)
    assert headers["X-Lobes-Tier-Reason"] == "busy"
    body = json.loads(resp.body)
    assert body["error"]["type"] == "server_busy"
    assert body["error"]["code"] == "busy"


def test_cortex_request_after_pressure_clears_gets_200_from_real_cortex() -> None:
    """Transient-busy narrative: 429 under pressure, then 200 after it clears.

    First call model=cortex under HIGH pressure → 429 (busy).  Then the same
    model=cortex with CLEARED pressure → 200 from the PRIMARY (cortex) backend,
    proving the busy signal is transient and a retrying client eventually gets
    a genuine cortex answer.
    """  # noqa: E501
    table, cfg = _fleet_cfg()

    # Phase 1: under pressure → shed
    opener1, calls1 = _opener({"hand": 200, "multimodal": 200, "primary": 200})
    resp1 = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"cortex"}', opener1, pressure=_HIGH_SWAP
    )
    assert resp1.status == 429
    assert calls1 == []

    # Phase 2: pressure clears → served from the real cortex (primary) backend
    opener2, calls2 = _opener({"hand": 200, "multimodal": 200, "primary": 200})
    resp2 = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [],
        b'{"model":"cortex"}',
        opener2,
        pressure=_NO_PRESSURE,
    )
    assert resp2.status == 200
    assert resp2.upstream is not None
    # The primary (cortex) backend was dialed, not a substitute.
    assert calls2[0][0] == "primary"
    # The forwarded body's model was rewritten to the primary served name.
    fwd = json.loads(calls2[0][1])
    assert fwd["model"] == "PRIMARY"
    headers = dict(resp2.headers)
    assert headers["X-Lobes-Tier"] == "main"
    assert headers["X-Lobes-Tier-Reason"] == "default"


def test_minor_request_served_even_under_pressure() -> None:
    """model=minor under HIGH pressure → 200, dialed the minor backend.

    The floor tier is never shed.
    """
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"hand": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"minor"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 200
    assert resp.upstream is not None
    assert calls[0][0] == "hand"  # served, not shed
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "hand"
    assert headers["X-Lobes-Tier-Reason"] == "default"


# --- thermal shed (lane-safety t14) -----------------------------------------

import pytest  # noqa: E402

from lobes.gateway import _pressure_policy as P  # noqa: E402


@pytest.fixture
def spark_band(monkeypatch):
    """The spark card's declared band: shed above 91 C, resume below 86 C."""
    monkeypatch.setattr(P, "THERMAL_SHED_C", 91.0)
    monkeypatch.setattr(P, "THERMAL_RESUME_C", 86.0)


def _thermal(t, latched=False):
    return {
        "swap_used_percent": 0.0,
        "iowait_percent": 0.0,
        "thermal_c": t,
        "thermal_latched": latched,
    }


def _post(model, pressure, *, headers=(), behavior=None):
    table, cfg = _fleet_cfg()
    opener, calls = _opener(behavior or {"hand": 200, "multimodal": 200, "primary": 200})
    body = json.dumps({"model": model}).encode()
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", list(headers), body, opener, pressure=pressure
    )
    return resp, calls


def test_hot_box_sheds_cortex_with_the_thermal_reason(spark_band) -> None:
    resp, calls = _post("cortex", _thermal(93.0))
    assert resp.status == 429
    assert calls == []
    headers = dict(resp.headers)
    assert headers["Retry-After"] == str(S.BUSY_RETRY_AFTER_SECONDS)
    assert headers["X-Lobes-Tier-Reason"] == "busy"  # busy semantics unchanged
    assert headers[S.SHED_SIGNAL_HEADER] == "thermal"
    err = json.loads(resp.body)["error"]
    assert err["type"] == "server_busy"
    assert err["code"] == "busy"
    assert err["shed_signal"] == "thermal"


def test_swap_shed_carries_no_shed_signal_header_byte_identity(spark_band) -> None:
    resp, _ = _post("cortex", _HIGH_SWAP)
    assert resp.status == 429
    assert S.SHED_SIGNAL_HEADER not in dict(resp.headers)
    assert "shed_signal" not in json.loads(resp.body)["error"]


def test_hand_is_served_on_a_hot_box(spark_band) -> None:
    resp, calls = _post("hand", _thermal(99.0, latched=True))
    assert resp.status == 200
    assert calls[0][0] == "hand"


def test_override_bypasses_the_thermal_shed(spark_band) -> None:
    # The handler turns `X-Lobes-Override` into handle_post's override flag.
    assert S.is_override("1") is True
    table, cfg = _fleet_cfg()
    opener, calls = _opener({"hand": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [],
        b'{"model":"cortex"}',
        opener,
        pressure=_thermal(99.0, latched=True),
        override=True,
    )
    assert resp.status == 200
    assert calls[0][0] == "primary"
    assert dict(resp.headers)["X-Lobes-Tier-Reason"] == "manual_override"


def test_unknown_temperature_never_sheds(spark_band) -> None:
    resp, calls = _post("cortex", _thermal(None, latched=True))
    assert resp.status == 200
    assert calls[0][0] == "primary"


def test_hysteresis_reaches_the_wire(spark_band) -> None:
    assert _post("cortex", _thermal(88.0))[0].status == 200
    assert _post("cortex", _thermal(88.0, latched=True))[0].status == 429
    assert _post("cortex", _thermal(85.0, latched=True))[0].status == 200


class _FixedCache:
    def __init__(self, value):
        self.value = value

    def current(self):
        return dict(self.value)


def test_handler_latch_carries_hysteresis_across_requests(spark_band) -> None:
    table, cfg = _fleet_cfg()
    cache = _FixedCache({"swap_used_percent": 0.0, "iowait_percent": 0.0, "thermal_c": 92.0})
    handler_cls = S._make_handler(table, cfg, pressure_cache=cache)
    handler = handler_cls.__new__(handler_cls)  # no socket: only the pressure read
    assert handler._current_pressure()["thermal_latched"] is True
    cache.value["thermal_c"] = 88.0  # inside the hysteresis band → still hot
    assert handler._current_pressure()["thermal_latched"] is True
    cache.value["thermal_c"] = 85.0
    assert handler._current_pressure()["thermal_latched"] is False
    cache.value["thermal_c"] = 88.0  # climbing back, not past shed → cool
    assert handler._current_pressure()["thermal_latched"] is False
    # Two servers never share a latch.
    other = S._make_handler(table, cfg, pressure_cache=cache)
    assert other.thermal_latch is not handler_cls.thermal_latch


def test_handler_without_a_pressure_cache_reads_none() -> None:
    table, cfg = _fleet_cfg()
    handler_cls = S._make_handler(table, cfg)
    assert handler_cls.__new__(handler_cls)._current_pressure() is None


def _fake_probe(_url, timeout):  # noqa: ARG001
    return {"health": "ok", "metrics": {"running": 0, "waiting": 0}}


def test_status_reports_thermal_reading_and_signal(spark_band) -> None:
    from lobes.gateway._replicas import _busy_from_status

    table, cfg = _fleet_cfg()
    payload = S.fleet_status_payload(table, cfg, _thermal(88.0, latched=True), probe=_fake_probe)
    pressure = payload["pressure"]
    assert pressure["thermal_c"] == 88.0
    assert pressure["thermal_shedding"] is True
    assert pressure["shed_signal"] == "thermal"
    assert pressure["shed"] is True
    assert pressure["mode"] == "busy"
    # A peer reading this /status counts the box as busy.
    assert _busy_from_status(payload) is True

    cool = S.fleet_status_payload(table, cfg, _thermal(None), probe=_fake_probe)["pressure"]
    assert cool["thermal_c"] is None
    assert cool["thermal_shedding"] is False
    assert cool["shed_signal"] == ""
    assert _busy_from_status({"pressure": cool}) is False


def test_status_without_a_thermal_key_is_unchanged() -> None:
    table, cfg = _fleet_cfg()
    pressure = S.fleet_status_payload(table, cfg, _NO_PRESSURE, probe=_fake_probe)["pressure"]
    assert set(pressure) == {"mode", "shed", "reason", "swap_used_percent", "iowait_percent"}
