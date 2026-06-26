"""Tests for per-lobe perf engine (t3): ttft, concurrent driver, auto-ramp knee.

All tests are hermetic — no real GPU or network.  The network tests use a real
``http.server.ThreadingHTTPServer`` on ``127.0.0.1:0`` (OS picks the port) that
returns a canned chat-completions payload with a small sleep so latency is
measurable.  Ramp / knee tests inject a fake measurement function and need no
sockets at all.
"""

from __future__ import annotations

import http.server
import json
import threading
import time

import pytest

import lobes.assess as A

# ---------------------------------------------------------------------------
# Canned HTTP server (concurrent-capable via ThreadingHTTPServer)
# ---------------------------------------------------------------------------

_CANNED_COMPLETION_TOKENS = 10
_CANNED_PROMPT_TOKENS = 100
_SLEEP_S = 0.01  # small but measurable latency


class _PerfHandler(http.server.BaseHTTPRequestHandler):
    """Serve canned chat-completions responses with a tiny sleep."""

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        time.sleep(_SLEEP_S)
        body = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": _CANNED_PROMPT_TOKENS,
                    "completion_tokens": _CANNED_COMPLETION_TOKENS,
                },
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass  # silence log noise in test output


@pytest.fixture()
def perf_server():
    """ThreadingHTTPServer on an ephemeral port; yields the base URL string."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _PerfHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()


# ---------------------------------------------------------------------------
# Acceptance criterion 1 — measure_prefill_ttft
# ---------------------------------------------------------------------------


def test_measure_prefill_ttft_returns_required_keys(perf_server: str) -> None:
    """measure_prefill_ttft returns both ttft_ms and prompt_tokens."""
    result = A.measure_prefill_ttft(perf_server, "test-model")
    assert "ttft_ms" in result, "missing key: ttft_ms"
    assert "prompt_tokens" in result, "missing key: prompt_tokens"


def test_measure_prefill_ttft_ttft_positive(perf_server: str) -> None:
    """ttft_ms is a positive float (round-trip time > 0)."""
    result = A.measure_prefill_ttft(perf_server, "test-model")
    assert isinstance(result["ttft_ms"], float)
    assert result["ttft_ms"] > 0.0


def test_measure_prefill_ttft_prompt_tokens_positive(perf_server: str) -> None:
    """prompt_tokens is a positive int (server echoes _CANNED_PROMPT_TOKENS=100)."""
    result = A.measure_prefill_ttft(perf_server, "test-model")
    assert isinstance(result["prompt_tokens"], int)
    assert result["prompt_tokens"] > 0


def test_measure_prefill_ttft_custom_input_len(perf_server: str) -> None:
    """Passing a custom input_len still returns valid keys with positive values."""
    result = A.measure_prefill_ttft(perf_server, "test-model", input_len=500)
    assert result["ttft_ms"] > 0.0
    assert result["prompt_tokens"] > 0


# ---------------------------------------------------------------------------
# Acceptance criterion 2 — run_concurrent
# ---------------------------------------------------------------------------


def test_run_concurrent_returns_all_keys(perf_server: str) -> None:
    """run_concurrent returns every required key."""
    result = A.run_concurrent(perf_server, "test-model", concurrency=4)
    for key in (
        "concurrency",
        "requests_per_s",
        "p50_latency_ms",
        "p95_latency_ms",
        "ms_per_token",
        "total_s",
    ):
        assert key in result, f"missing key: {key}"


def test_run_concurrent_p95_ge_p50(perf_server: str) -> None:
    """p95_latency_ms is always >= p50_latency_ms (sorted-percentile guarantee)."""
    result = A.run_concurrent(perf_server, "test-model", concurrency=4)
    assert result["p95_latency_ms"] >= result["p50_latency_ms"]


def test_run_concurrent_requests_per_s_positive(perf_server: str) -> None:
    """requests_per_s > 0."""
    result = A.run_concurrent(perf_server, "test-model", concurrency=4)
    assert result["requests_per_s"] > 0.0


def test_run_concurrent_concurrency_field(perf_server: str) -> None:
    """The concurrency field echoes the requested concurrency."""
    result = A.run_concurrent(perf_server, "test-model", concurrency=4)
    assert result["concurrency"] == 4


def test_run_concurrent_ms_per_token_positive(perf_server: str) -> None:
    """ms_per_token > 0 (server always returns completion_tokens > 0)."""
    result = A.run_concurrent(perf_server, "test-model", concurrency=2)
    assert result["ms_per_token"] > 0.0


def test_run_concurrent_total_s_positive(perf_server: str) -> None:
    """total_s > 0 (elapsed wall time for the batch)."""
    result = A.run_concurrent(perf_server, "test-model", concurrency=2)
    assert result["total_s"] > 0.0


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — _find_knee (pure unit tests, no network)
# ---------------------------------------------------------------------------


def test_find_knee_stops_at_plateau() -> None:
    """Gain drops below threshold → knee is the last concurrency before the drop."""
    rows = [
        {"concurrency": 1, "requests_per_s": 10.0},
        {"concurrency": 2, "requests_per_s": 19.0},  # gain = 0.90 > 0.10 → continue
        {"concurrency": 4, "requests_per_s": 21.0},  # gain = 0.105 > 0.10 → continue
        {"concurrency": 8, "requests_per_s": 21.5},  # gain = 0.024 < 0.10 → STOP
    ]
    result = A._find_knee(rows, threshold=0.1)
    assert result["knee"] == 4  # row[2].concurrency — last before gain dropped
    assert len(result["rows"]) == 3  # rows[:3], NOT including the declining step


def test_find_knee_keeps_climbing() -> None:
    """All steps gain >= threshold → knee is the last step; all rows returned."""
    rows = [
        {"concurrency": 1, "requests_per_s": 10.0},
        {"concurrency": 2, "requests_per_s": 20.0},  # gain = 1.0 > 0.10
        {"concurrency": 4, "requests_per_s": 35.0},  # gain = 0.75 > 0.10
    ]
    result = A._find_knee(rows, threshold=0.1)
    assert result["knee"] == 4
    assert len(result["rows"]) == 3


def test_find_knee_single_row() -> None:
    """A single row has no comparison — knee is its concurrency, rows echoes it."""
    rows = [{"concurrency": 1, "requests_per_s": 10.0}]
    result = A._find_knee(rows, threshold=0.1)
    assert result["knee"] == 1
    assert result["rows"] == rows


def test_find_knee_first_step_triggers() -> None:
    """Gain drops immediately at step 2 → knee is step 1, rows contains only step 1."""
    rows = [
        {"concurrency": 1, "requests_per_s": 10.0},
        {"concurrency": 2, "requests_per_s": 10.5},  # gain = 0.05 < 0.10 → STOP
    ]
    result = A._find_knee(rows, threshold=0.1)
    assert result["knee"] == 1
    assert len(result["rows"]) == 1
    assert result["rows"][0]["concurrency"] == 1


def test_find_knee_exact_threshold_boundary() -> None:
    """Gain exactly equal to threshold does NOT stop (strict < comparison)."""
    rows = [
        {"concurrency": 1, "requests_per_s": 10.0},
        {"concurrency": 2, "requests_per_s": 11.0},  # gain = 0.10 — NOT < 0.10
        {"concurrency": 4, "requests_per_s": 11.05},  # gain ≈ 0.0045 < 0.10 → STOP
    ]
    result = A._find_knee(rows, threshold=0.1)
    # gain at step 2 is exactly 0.1 which is NOT < 0.1, so we continue
    assert result["knee"] == 2
    assert len(result["rows"]) == 2


def test_find_knee_empty_rows() -> None:
    """Empty rows list returns knee=0 and empty rows (degenerate case)."""
    result = A._find_knee([], threshold=0.1)
    assert result["knee"] == 0
    assert result["rows"] == []


# ---------------------------------------------------------------------------
# Acceptance criterion 3 — auto_ramp_concurrency (injected fake, no sockets)
# ---------------------------------------------------------------------------


def _make_fake_measure(throughputs: dict[int, float]) -> tuple[object, list[int]]:
    """Return (fake_measure, calls_list) where calls_list records concurrencies used."""
    calls: list[int] = []

    def fake_measure(url: str, model: str, *, concurrency: int, **kw: object) -> dict:
        calls.append(concurrency)
        rps = throughputs[concurrency]
        return {
            "concurrency": concurrency,
            "requests_per_s": rps,
            "p50_latency_ms": 10.0,
            "p95_latency_ms": 15.0,
            "ms_per_token": 1.0,
            "total_s": concurrency / rps,
        }

    return fake_measure, calls


def test_auto_ramp_stops_at_knee() -> None:
    """Injected fake: ramp stops early at the knee; higher concurrencies not measured."""
    # gain(1→2) = 0.8 ≥ 0.1 → continue
    # gain(2→4) = (18.5-18)/18 ≈ 0.028 < 0.1 → STOP (c=8 with rps=100 never reached)
    throughputs = {1: 10.0, 2: 18.0, 4: 18.5, 8: 100.0}
    fake, calls = _make_fake_measure(throughputs)

    result = A.auto_ramp_concurrency(
        "http://ignored",
        "test-model",
        schedule=(1, 2, 4, 8),
        threshold=0.1,
        _measure=fake,
    )

    assert result["knee"] == 2  # last concurrency before gain dropped
    assert len(result["rows"]) == 2
    assert result["rows"][-1]["concurrency"] == 2
    assert 8 not in calls, "concurrency=8 should never have been measured"


def test_auto_ramp_all_climbing() -> None:
    """When all steps gain >= threshold, knee is the last step; all rows returned."""
    # gain(1→2)=1.0, gain(2→4)=0.9 — both well above 0.1
    throughputs = {1: 10.0, 2: 20.0, 4: 38.0}
    fake, calls = _make_fake_measure(throughputs)

    result = A.auto_ramp_concurrency(
        "http://ignored",
        "test-model",
        schedule=(1, 2, 4),
        threshold=0.1,
        _measure=fake,
    )

    assert result["knee"] == 4
    assert len(result["rows"]) == 3
    assert calls == [1, 2, 4]


def test_auto_ramp_returns_per_step_rows() -> None:
    """result['rows'] contains per-step dicts with at least concurrency + requests_per_s."""
    throughputs = {1: 10.0, 2: 19.0, 4: 19.5}
    fake, _calls = _make_fake_measure(throughputs)

    result = A.auto_ramp_concurrency(
        "http://ignored",
        "test-model",
        schedule=(1, 2, 4),
        threshold=0.1,
        _measure=fake,
    )

    for row in result["rows"]:
        assert "concurrency" in row
        assert "requests_per_s" in row


def test_auto_ramp_knee_single_step() -> None:
    """With a one-element schedule, knee = that concurrency and rows has 1 entry."""
    throughputs = {1: 10.0}
    fake, calls = _make_fake_measure(throughputs)

    result = A.auto_ramp_concurrency(
        "http://ignored",
        "test-model",
        schedule=(1,),
        threshold=0.1,
        _measure=fake,
    )

    assert result["knee"] == 1
    assert len(result["rows"]) == 1
    assert calls == [1]
