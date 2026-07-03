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
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MINOR_SERVED_NAME": "MINOR",
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
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
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
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
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
    opener1, calls1 = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp1 = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"cortex"}', opener1, pressure=_HIGH_SWAP
    )
    assert resp1.status == 429
    assert calls1 == []

    # Phase 2: pressure clears → served from the real cortex (primary) backend
    opener2, calls2 = _opener({"minor": 200, "multimodal": 200, "primary": 200})
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
    opener, calls = _opener({"minor": 200, "multimodal": 200, "primary": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], b'{"model":"minor"}', opener, pressure=_HIGH_SWAP
    )
    assert resp.status == 200
    assert resp.upstream is not None
    assert calls[0][0] == "minor"  # served, not shed
    headers = dict(resp.headers)
    assert headers["X-Lobes-Tier"] == "minor"
    assert headers["X-Lobes-Tier-Reason"] == "default"
