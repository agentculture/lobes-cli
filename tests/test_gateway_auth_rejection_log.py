"""What the log tells the operator when the inbound gate rejects (issue #228).

The gate itself was never the problem — a 401 on an unauthenticated POST is
#127 working. What it *said* was: measured on the DGX Spark 2026-08-28/29,
**1190 rejections in four hours**, each rendered as

    [gateway] "POST /tokenize HTTP/1.1" 401 -

naming neither the source nor the reason, and repeated verbatim 1190 times.
That log cannot answer "one misconfigured client, or a scan?", and it buries
everything else the gateway had to say in the same window.

Two halves are pinned here: the collapse policy (pure, clock-injected, no
sockets) and the handler wiring end-to-end through a real loopback server.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lobes.gateway import _authlog
from lobes.gateway import server as S
from lobes.gateway._authlog import RejectionLog, rejection_reason

# --- the collapse policy ----------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def _log(clock: _Clock, **kw) -> RejectionLog:
    return RejectionLog(clock=clock, **kw)


def test_the_first_rejection_from_a_source_is_always_reported() -> None:
    clock = _Clock()
    line = _log(clock).record("172.21.0.1", "POST", "/tokenize", _authlog.REASON_NO_HEADER)
    assert line == ("auth: rejected POST /tokenize from 172.21.0.1 (no Authorization header)")


def test_a_burst_from_one_source_collapses_to_a_single_line() -> None:
    """The measured shape: the #228 resident retried in bursts of ~29."""
    clock = _Clock()
    log = _log(clock, window=60.0)
    lines = []
    for _ in range(29):
        clock.now += 0.1
        lines.append(log.record("172.21.0.1", "POST", "/tokenize", _authlog.REASON_NO_HEADER))
    assert sum(line is not None for line in lines) == 1


def test_the_next_window_reports_and_carries_the_suppressed_count() -> None:
    """A sustained flood must stay visible as a rising number — neither
    vanishing after the first line nor drowning the log."""
    clock = _Clock()
    log = _log(clock, window=60.0)
    for _ in range(29):
        clock.now += 0.1
        log.record("172.21.0.1", "POST", "/tokenize", _authlog.REASON_NO_HEADER)
    clock.now += 120.0
    line = log.record("172.21.0.1", "POST", "/tokenize", _authlog.REASON_NO_HEADER)
    assert line is not None
    assert "[+28 more from this source in the previous 123s]" in line


def test_a_lone_rejection_reads_as_one_plain_sentence() -> None:
    """No suppressed count when nothing was suppressed — the common case (a
    single stray request) must not carry flood machinery."""
    clock = _Clock()
    log = _log(clock, window=60.0)
    log.record("10.0.0.9", "POST", "/v1/chat/completions", _authlog.REASON_MISMATCH)
    clock.now += 120.0
    assert "[+" not in log.record("10.0.0.9", "POST", "/v1/chat/completions", "x")


def test_sources_collapse_independently() -> None:
    """One noisy source must never silence a different one — that would hide
    the very escalation (a second address appearing) worth alerting on."""
    clock = _Clock()
    log = _log(clock, window=60.0)
    assert log.record("10.0.0.1", "POST", "/tokenize", "r") is not None
    assert log.record("10.0.0.1", "POST", "/tokenize", "r") is None
    assert log.record("10.0.0.2", "POST", "/tokenize", "r") is not None


def test_the_tracking_table_is_bounded_against_a_distributed_scan() -> None:
    """The keys are attacker-chosen, so an unbounded dict would turn a scan
    into a memory leak in the process being scanned."""
    clock = _Clock()
    log = _log(clock, window=60.0, max_sources=8)
    for i in range(500):
        clock.now += 0.01
        log.record(f"10.0.{i // 256}.{i % 256}", "POST", "/tokenize", "r")
    assert len(log._sources) <= 8


def test_eviction_prefers_expired_windows_over_live_ones() -> None:
    clock = _Clock()
    log = _log(clock, window=10.0, max_sources=2)
    log.record("stale", "POST", "/x", "r")
    clock.now += 20.0  # 'stale' window has expired
    log.record("live", "POST", "/x", "r")
    log.record("newest", "POST", "/x", "r")
    assert "stale" not in log._sources
    assert "live" in log._sources


# --- the reason classification ---------------------------------------------


@pytest.mark.parametrize(
    "header,expected",
    [
        (None, _authlog.REASON_NO_HEADER),
        ("", _authlog.REASON_NO_HEADER),
        ("Basic dXNlcjpwYXNz", _authlog.REASON_NOT_BEARER),
        ("some-bare-token", _authlog.REASON_NOT_BEARER),
        ("Bearer ", _authlog.REASON_EMPTY_TOKEN),
        ("Bearer   ", _authlog.REASON_EMPTY_TOKEN),
        ("Bearer the-wrong-key", _authlog.REASON_MISMATCH),
        ("bearer the-wrong-key", _authlog.REASON_MISMATCH),  # scheme is case-insensitive
    ],
)
def test_reason_names_the_step_that_actually_failed(header, expected) -> None:
    assert rejection_reason(header) == expected


def test_the_reason_never_contains_the_presented_credential() -> None:
    """The classification must not become the leak the static 401 body avoids."""
    secret = "sk-super-secret-token-value"
    for header in (f"Bearer {secret}", f"Basic {secret}", secret):
        assert secret not in rejection_reason(header)


def test_reason_order_mirrors_the_matcher_it_explains() -> None:
    """A category that named a different step than the one that failed would
    send the operator after the wrong fix."""
    assert rejection_reason("Basic ") == _authlog.REASON_NOT_BEARER  # scheme checked first
    assert S.bearer_token_matches("k", "Basic ") is False


# --- the handler wiring, end to end ----------------------------------------

_KEY = "test-inbound-key"


class _FakeUpstream:
    """A 200 with a tiny buffered body — no socket, no backend."""

    status = 200
    headers = [("Content-Type", "application/json")]

    def read_all(self) -> bytes:
        return b'{"ok": true}'

    def read(self, _n: int) -> bytes:
        return b""

    def close(self) -> None:
        return None


@pytest.fixture
def logging_gateway(monkeypatch):
    """A real loopback gateway with the inbound key set, mirroring
    tests/test_gateway_auth.py's fixture conventions."""
    from lobes.gateway._config import build_config

    table, cfg = build_config(
        {
            "PRIMARY_URL": "http://vllm-primary:8000",
            "PRIMARY_SERVED_NAME": "some/cortex",
            "GATEWAY_API_KEY": _KEY,
        }
    )
    dialed: list[str] = []

    def fake_open(backend, path, body, headers, **_kw):
        dialed.append(f"{backend.name}{path}")
        return _FakeUpstream()

    monkeypatch.setattr(S, "open_upstream", fake_open)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    try:
        yield SimpleNamespace(base=f"http://{host}:{port}", httpd=httpd, dialed=dialed)
    finally:
        httpd.shutdown()
        httpd.server_close()


def _post(base: str, path: str, header: str | None = None) -> int:
    req = urllib.request.Request(
        base + path, data=json.dumps({"model": "some/cortex"}).encode(), method="POST"
    )
    if header is not None:
        req.add_header("Authorization", header)
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_a_rejection_logs_source_reason_and_path(logging_gateway, capfd) -> None:
    """The three facts the 1190-line flood was missing."""
    assert _post(logging_gateway.base, "/tokenize") == 401
    err = capfd.readouterr().err
    assert "auth: rejected POST /tokenize" in err
    assert "from 127.0.0.1" in err
    assert _authlog.REASON_NO_HEADER in err


def test_a_burst_through_the_real_server_collapses_both_log_lines(logging_gateway, capfd) -> None:
    """Suppressing the diagnostic but keeping the access line would have left
    the observed flood at its observed size — so BOTH must go."""
    for _ in range(25):
        assert _post(logging_gateway.base, "/tokenize") == 401
    err = capfd.readouterr().err
    assert err.count("auth: rejected") == 1
    # ...and the ordinary access line is suppressed for the collapsed ones too.
    assert err.count('"POST /tokenize HTTP/1.1" 401') == 1


def test_the_401_response_still_says_nothing_the_log_now_says(logging_gateway, capfd) -> None:
    """The asymmetry is the point: the operator learns the reason, the caller
    never does — a 401 must not become a key-material oracle."""
    secret = "sk-a-wrong-but-secret-looking-key"
    req = urllib.request.Request(
        logging_gateway.base + "/v1/chat/completions", data=b"{}", method="POST"
    )
    req.add_header("Authorization", f"Bearer {secret}")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(req, timeout=5)
    body = caught.value.read().decode()
    payload = json.loads(body)
    assert payload["error"]["code"] == "invalid_api_key"
    # The response distinguishes nothing and echoes nothing...
    for reason in (
        _authlog.REASON_NO_HEADER,
        _authlog.REASON_NOT_BEARER,
        _authlog.REASON_EMPTY_TOKEN,
        _authlog.REASON_MISMATCH,
    ):
        assert reason not in body
    assert secret not in body
    assert _KEY not in body
    # ...while the LOG names the reason, and still never the key material.
    err = capfd.readouterr().err
    assert _authlog.REASON_MISMATCH in err
    assert secret not in err
    assert _KEY not in err


def test_an_authorized_request_logs_no_rejection_line(logging_gateway, capfd) -> None:
    """A successful request must not pay for this machinery."""
    assert _post(logging_gateway.base, "/v1/chat/completions", f"Bearer {_KEY}") == 200
    assert logging_gateway.dialed == ["primary/v1/chat/completions"]
    assert "auth: rejected" not in capfd.readouterr().err


def test_a_rejected_request_never_reaches_a_backend(logging_gateway, capfd) -> None:
    """The #127 property this must not have disturbed: the gate runs before
    any body parse, model resolution or upstream socket."""
    for _ in range(5):
        assert _post(logging_gateway.base, "/tokenize") == 401
    assert logging_gateway.dialed == []


def test_tokenize_is_served_by_path_passthrough_when_authorized(logging_gateway) -> None:
    """Why #228's traffic existed at all: `/tokenize` is a vLLM-native route
    the gateway has no case for, but `handle_post` is path-agnostic — it
    resolves the model from the body and forwards the ORIGINAL path to the
    owning lane, which does serve it. So the 401s were never "no such route";
    they were a client with no key on a route that works."""
    assert _post(logging_gateway.base, "/tokenize", f"Bearer {_KEY}") == 200
    assert logging_gateway.dialed == ["primary/tokenize"]


def test_a_handler_without_a_rejection_log_still_rejects_and_logs_plainly() -> None:
    """`rejection_log` is None on a hand-built handler; the gate must not
    depend on the logging that decorates it."""
    handler = S._Handler.__new__(S._Handler)
    handler.rejection_log = None
    handler.headers = {}
    assert handler._log_rejection() is False
