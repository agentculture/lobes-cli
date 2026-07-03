"""Tests for ``lobes measure`` (issue #81, task t8) — per-role RUNTIME metrics.

Covers both layers:

* :mod:`lobes.roles_measure` — the pure probe logic, exercised against a real
  (but tiny, local, ephemeral-port) HTTP server for the "reachable" path and a
  closed port (nothing listening) for the "unreachable" path — no live models,
  no mocking framework needed for the network shape itself.
* ``lobes measure`` (the CLI verb) — deployment resolution, ``--json``/table
  rendering, ``--role`` filtering, and the read-only / RUNTIME-ONLY contract.

No live model is ever required: every "reachable" test talks to a canned
local server; every "unreachable" test talks to a closed OS port (instant
connection-refused, no real timeout wait).
"""

from __future__ import annotations

import http.server
import io
import json
import socket
import threading
import time
import wave

import pytest

from lobes import roles_measure as RM
from lobes.cli import main
from lobes.roles import ROLES, RoleInfo
from lobes.runtime import _compose

# ---------------------------------------------------------------------------
# Fixtures — a canned local server (all six roles' wire shapes) + a closed port
# ---------------------------------------------------------------------------

_CANNED_PROMPT_TOKENS = 100
_CANNED_COMPLETION_TOKENS = 10
_SLEEP_S = 0.005  # small but measurable latency, mirrors tests/test_assess_perf.py

_VLLM_METRICS_TEXT = (
    'vllm:num_requests_running{model_name="x"} 1\n'
    'vllm:num_requests_waiting{model_name="x"} 0\n'
    'vllm:gpu_cache_usage_perc{model_name="x"} 0.42\n'
)


def _canned_wav_bytes(duration_s: float = 0.2, rate: int = 24000) -> bytes:
    n_frames = int(duration_s * rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * n_frames)
    return buf.getvalue()


class _MeasureHandler(http.server.BaseHTTPRequestHandler):
    """Canned responses for every wire shape ``lobes.roles_measure`` probes."""

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_bytes(200, b"ok", "text/plain")
            return
        if self.path == "/metrics":
            self._write_bytes(200, _VLLM_METRICS_TEXT.encode(), "text/plain")
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        time.sleep(_SLEEP_S)
        if self.path == "/v1/chat/completions":
            self._write_json(
                200,
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": _CANNED_PROMPT_TOKENS,
                        "completion_tokens": _CANNED_COMPLETION_TOKENS,
                    },
                },
            )
            return
        if self.path == "/v1/embeddings":
            self._write_json(200, {"data": [{"embedding": [0.1, 0.2]}], "model": "x"})
            return
        if self.path == "/v1/rerank":
            self._write_json(200, {"results": [{"index": 0, "relevance_score": 0.9}]})
            return
        if self.path == "/v1/audio/speech":
            self._write_bytes(200, _canned_wav_bytes(), "audio/wav")
            return
        if self.path == "/v1/audio/transcriptions":
            self._write_json(200, {"text": "hello"})
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass  # silence log noise in test output


@pytest.fixture()
def measure_server():
    """``ThreadingHTTPServer`` on an ephemeral port; yields the base URL string."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _MeasureHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _closed_port() -> int:
    """A port nobody is listening on — connections refuse instantly, no timeout wait."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _role_info(
    role: str,
    *,
    endpoint: str,
    loaded: bool = True,
    model: str = "test-model",
    context: int = 4096,
    runtime: str = "vllm",
) -> RoleInfo:
    return RoleInfo(
        role=role,
        model=model,
        runtime=runtime,
        endpoint=endpoint,
        path="/v1/chat/completions",
        context=context,
        quant="",
        mtp=False,
        responsibilities=("x",),
        forbidden_responsibilities=(),
        ready=None,
        loaded=loaded,
    )


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


# ---------------------------------------------------------------------------
# Acceptance 3 — RUNTIME-ONLY vocabulary (boundary c7/h14)
# ---------------------------------------------------------------------------

_FORBIDDEN_SUBSTRINGS = (
    "correct",
    "accuracy",
    "quality",
    "score",
    "success",
    "valid",
)


def test_allowed_metric_keys_never_assert_correctness_or_quality() -> None:
    for key in RM.ALLOWED_METRIC_KEYS:
        lowered = key.lower()
        for bad in _FORBIDDEN_SUBSTRINGS:
            assert bad not in lowered, f"{key!r} reads like a correctness/quality claim"


def test_allowed_metric_keys_is_the_union_of_the_three_families() -> None:
    assert RM.LLM_METRIC_KEYS
    assert RM.EMBED_RERANK_METRIC_KEYS
    assert RM.AUDIO_METRIC_KEYS
    assert RM.ALLOWED_METRIC_KEYS == (
        RM.LLM_METRIC_KEYS | RM.EMBED_RERANK_METRIC_KEYS | RM.AUDIO_METRIC_KEYS
    )


# ---------------------------------------------------------------------------
# lobes.roles_measure — LLM family (cortex/senses)
# ---------------------------------------------------------------------------


def test_llm_role_measured_when_reachable(measure_server: str) -> None:
    info = _role_info("cortex", endpoint=measure_server, context=4096)
    result = RM._measure_llm_role(info, timeout=5.0)
    assert result["ready"] is True
    m = result["metrics"]
    assert set(m) == set(RM.LLM_METRIC_KEYS)
    assert m["context"] == 4096
    assert m["ttft_ms"] > 0.0
    assert m["decode_tps"] > 0.0
    assert m["prefill_tps"] > 0.0
    assert m["mem_usage_pct"] == pytest.approx(0.42)
    # Not cheaply available without a docker inspect — always null, never invented.
    assert m["restart_count"] is None
    assert m["error_count"] is None


def test_llm_role_unreachable_degrades_gracefully() -> None:
    port = _closed_port()
    info = _role_info("cortex", endpoint=f"http://127.0.0.1:{port}", context=4096)
    result = RM._measure_llm_role(info, timeout=2.0)
    assert result["ready"] is False
    m = result["metrics"]
    assert m["context"] == 4096  # known without a probe — never nulled
    assert m["ttft_ms"] is None
    assert m["decode_tps"] is None
    assert m["prefill_tps"] is None
    assert m["mem_usage_pct"] is None


def test_llm_role_not_loaded_never_touches_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not probe the network when the role isn't loaded")

    monkeypatch.setattr(RM._metrics, "probe_backend", boom)
    monkeypatch.setattr(RM._assess, "measure_prefill_ttft", boom)
    monkeypatch.setattr(RM._assess, "_post", boom)
    info = _role_info("senses", endpoint="", loaded=False, context=32768)
    result = RM._measure_llm_role(info, timeout=2.0)
    assert result["ready"] is False
    assert result["metrics"]["context"] == 32768
    assert all(v is None for k, v in result["metrics"].items() if k != "context")


# ---------------------------------------------------------------------------
# lobes.roles_measure — pooling family (embedder/reranker)
# ---------------------------------------------------------------------------


def test_embedder_role_measured_when_reachable(measure_server: str) -> None:
    info = _role_info("embedder", endpoint=measure_server)
    result = RM._measure_embed_rerank_role(info, timeout=5.0)
    assert result["ready"] is True
    m = result["metrics"]
    assert set(m) == set(RM.EMBED_RERANK_METRIC_KEYS)
    assert m["batch_size"] == len(RM._EMBED_PROBE_INPUT)
    assert m["latency_ms"] > 0.0
    assert m["reqs_per_sec"] > 0.0
    assert m["docs_per_sec"] > 0.0
    assert m["loaded"] is True


def test_reranker_role_measured_when_reachable(measure_server: str) -> None:
    info = _role_info("reranker", endpoint=measure_server)
    result = RM._measure_embed_rerank_role(info, timeout=5.0)
    assert result["ready"] is True
    assert result["metrics"]["batch_size"] == len(RM._RERANK_PROBE_DOCS)


def test_embed_rerank_role_unreachable_degrades_gracefully() -> None:
    port = _closed_port()
    info = _role_info("embedder", endpoint=f"http://127.0.0.1:{port}")
    result = RM._measure_embed_rerank_role(info, timeout=2.0)
    assert result["ready"] is False
    m = result["metrics"]
    assert m["latency_ms"] is None
    assert m["reqs_per_sec"] is None
    assert m["docs_per_sec"] is None
    assert m["batch_size"] is None
    # 'loaded' mirrors the registry's config fact, not live reachability.
    assert m["loaded"] is True


def test_embed_rerank_role_not_loaded_never_touches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not probe the network when the role isn't loaded")

    monkeypatch.setattr(RM._metrics, "health_ok", boom)
    monkeypatch.setattr(RM._assess, "_post", boom)
    info = _role_info("reranker", endpoint="", loaded=False)
    result = RM._measure_embed_rerank_role(info, timeout=2.0)
    assert result["ready"] is False
    assert result["metrics"]["loaded"] is False


# ---------------------------------------------------------------------------
# lobes.roles_measure — audio family (stt/tts)
# ---------------------------------------------------------------------------


def test_tts_role_measured_when_reachable(measure_server: str) -> None:
    info = _role_info("tts", endpoint=measure_server, runtime="chatterbox")
    result = RM._measure_tts_role(info, timeout=5.0)
    assert result["ready"] is True
    m = result["metrics"]
    assert set(m) == set(RM.AUDIO_METRIC_KEYS)
    assert m["latency_ms"] > 0.0
    assert m["duration_ms"] > 0.0
    assert m["rtf"] > 0.0
    assert m["failure_rate"] == 0.0


def test_stt_role_measured_when_reachable(measure_server: str) -> None:
    info = _role_info("stt", endpoint=measure_server, runtime="parakeet")
    result = RM._measure_stt_role(info, timeout=5.0)
    assert result["ready"] is True
    m = result["metrics"]
    assert set(m) == set(RM.AUDIO_METRIC_KEYS)
    assert m["latency_ms"] > 0.0
    assert m["duration_ms"] == pytest.approx(500.0, abs=1.0)
    assert m["rtf"] > 0.0
    assert m["failure_rate"] == 0.0


def test_tts_role_unreachable_degrades_gracefully() -> None:
    port = _closed_port()
    info = _role_info("tts", endpoint=f"http://127.0.0.1:{port}", runtime="chatterbox")
    result = RM._measure_tts_role(info, timeout=2.0)
    assert result["ready"] is False
    assert result["metrics"]["failure_rate"] == 1.0
    assert result["metrics"]["latency_ms"] is None


def test_stt_role_unreachable_degrades_gracefully() -> None:
    port = _closed_port()
    info = _role_info("stt", endpoint=f"http://127.0.0.1:{port}", runtime="parakeet")
    result = RM._measure_stt_role(info, timeout=2.0)
    assert result["ready"] is False
    assert result["metrics"]["failure_rate"] == 1.0
    assert result["metrics"]["latency_ms"] is None


def test_audio_roles_not_loaded_never_touch_the_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not probe the network when the role isn't loaded")

    monkeypatch.setattr(RM._metrics, "health_ok", boom)
    for role, runtime, fn in (
        ("stt", "parakeet", RM._measure_stt_role),
        ("tts", "chatterbox", RM._measure_tts_role),
    ):
        info = _role_info(role, endpoint="", loaded=False, runtime=runtime)
        result = fn(info, timeout=2.0)
        assert result["ready"] is False
        assert result["metrics"]["failure_rate"] is None  # never probed at all


# ---------------------------------------------------------------------------
# measure_role / measure_registry — the dispatch + wrapping layer
# ---------------------------------------------------------------------------


def test_measure_role_wraps_family_result_with_common_fields(measure_server: str) -> None:
    info = _role_info("cortex", endpoint=measure_server, model="acme/x", context=999)
    out = RM.measure_role("cortex", info, timeout=5.0)
    assert out["role"] == "cortex"
    assert out["family"] == "llm"
    assert out["model"] == "acme/x"
    assert out["runtime"] == "vllm"
    assert out["endpoint"] == measure_server
    assert out["loaded"] is True
    assert out["ready"] is True
    assert set(out["metrics"]) == set(RM.LLM_METRIC_KEYS)


def test_measure_role_family_assignment_covers_all_six_roles() -> None:
    assert RM._FAMILY_BY_ROLE == {
        "cortex": "llm",
        "senses": "llm",
        "embedder": "embed_rerank",
        "reranker": "embed_rerank",
        "stt": "audio",
        "tts": "audio",
    }
    assert set(RM._FAMILY_BY_ROLE) == set(ROLES)


def test_measure_registry_measures_only_requested_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_measure_role(
        role: str, info: RoleInfo, *, timeout: float = RM.DEFAULT_TIMEOUT
    ) -> dict:
        calls.append(role)
        return {
            "role": role,
            "family": "x",
            "model": "m",
            "runtime": "r",
            "endpoint": "",
            "loaded": False,
            "ready": False,
            "metrics": {},
        }

    monkeypatch.setattr(RM, "measure_role", fake_measure_role)
    registry = {r: _role_info(r, endpoint="", loaded=False) for r in ROLES}
    out = RM.measure_registry(registry, roles=("cortex", "stt"))
    assert set(out) == {"cortex", "stt"}
    assert calls == ["cortex", "stt"]


def test_measure_registry_defaults_to_all_six_roles(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_measure_role(
        role: str, info: RoleInfo, *, timeout: float = RM.DEFAULT_TIMEOUT
    ) -> dict:
        calls.append(role)
        return {
            "role": role,
            "family": "x",
            "model": "m",
            "runtime": "r",
            "endpoint": "",
            "loaded": False,
            "ready": False,
            "metrics": {},
        }

    monkeypatch.setattr(RM, "measure_role", fake_measure_role)
    registry = {r: _role_info(r, endpoint="", loaded=False) for r in ROLES}
    out = RM.measure_registry(registry)
    assert set(out) == set(ROLES)
    assert set(calls) == set(ROLES)


# ---------------------------------------------------------------------------
# CLI — lobes measure
# ---------------------------------------------------------------------------


def test_cli_measure_json_returns_all_six_roles_with_family_appropriate_keys(
    tmp_path, capsys
) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()  # deterministic "unreachable" — no live model required
    rc = main(["measure", "--compose-dir", str(tmp_path), "--port", str(port), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == set(ROLES)
    for role in ("cortex", "senses"):
        assert set(payload[role]["metrics"]) == set(RM.LLM_METRIC_KEYS)
    for role in ("embedder", "reranker"):
        assert set(payload[role]["metrics"]) == set(RM.EMBED_RERANK_METRIC_KEYS)
    for role in ("stt", "tts"):
        assert set(payload[role]["metrics"]) == set(RM.AUDIO_METRIC_KEYS)


def test_cli_measure_json_unreachable_roles_report_ready_false_not_exception(
    tmp_path, capsys
) -> None:
    """Acceptance 4: an unreachable role never raises — it reports degraded metrics."""
    _scaffold_fleet(tmp_path)
    port = _closed_port()
    rc = main(["measure", "--compose-dir", str(tmp_path), "--port", str(port), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Wired-but-unreachable LLM roles: loaded (config fact) stays true, ready
    # goes false; every metric is null except 'context' (known without a probe).
    for role in ("cortex", "senses"):
        metrics = payload[role]["metrics"]
        assert payload[role]["loaded"] is True
        assert payload[role]["ready"] is False
        assert metrics["context"] == payload[role]["metrics"]["context"]  # present, non-null
        assert all(v is None for k, v in metrics.items() if k != "context")
    # Wired-but-unreachable pooling roles: same story, except 'loaded' mirrors
    # the config fact directly in the metrics dict too (part of its vocabulary).
    for role in ("embedder", "reranker"):
        metrics = payload[role]["metrics"]
        assert payload[role]["loaded"] is True
        assert payload[role]["ready"] is False
        assert metrics["loaded"] is True
        assert all(v is None for k, v in metrics.items() if k != "loaded")
    # stt/tts are unwired in this scaffold (no --audio overlay) → both false.
    for role in ("stt", "tts"):
        assert payload[role]["loaded"] is False
        assert payload[role]["ready"] is False


def test_cli_measure_role_filter(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()
    rc = main(
        [
            "measure",
            "--role",
            "embedder",
            "--compose-dir",
            str(tmp_path),
            "--port",
            str(port),
            "--json",
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"embedder"}


def test_cli_measure_unknown_role_rejected_by_argparse(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["measure", "--role", "bogus"])
    assert exc.value.code == 1  # EXIT_USER_ERROR via the structured argparse error


def test_cli_measure_metric_keys_are_runtime_only_vocabulary(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()
    rc = main(["measure", "--compose-dir", str(tmp_path), "--port", str(port), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    for role in ROLES:
        assert set(payload[role]["metrics"]) <= RM.ALLOWED_METRIC_KEYS


def test_cli_measure_non_json_renders_readable_table(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()
    rc = main(["measure", "--compose-dir", str(tmp_path), "--port", str(port)])
    assert rc == 0
    out = capsys.readouterr().out
    for role in ROLES:
        assert role in out


# ---------------------------------------------------------------------------
# Acceptance 2 — READ-ONLY: no compose/docker mutation, ever
# ---------------------------------------------------------------------------


def test_cli_measure_never_touches_docker(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    port = _closed_port()

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("measure must never invoke docker/compose")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    monkeypatch.setattr(_compose, "compose_down", boom)
    monkeypatch.setattr(_compose, "_run", boom)
    monkeypatch.setattr(_compose, "_probe", boom)
    rc = main(["measure", "--compose-dir", str(tmp_path), "--port", str(port), "--json"])
    assert rc == 0


def test_cli_measure_has_no_apply_flag() -> None:
    """Read-only verb: no --apply, unlike switch/serve/stop/init/fleet/tunnel."""
    with pytest.raises(SystemExit) as exc:
        main(["measure", "--apply"])
    assert exc.value.code == 1


# ---------------------------------------------------------------------------
# Registration — shows up in --help / overview
# ---------------------------------------------------------------------------


def test_measure_appears_in_top_level_help(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "measure" in capsys.readouterr().out


def test_overview_lists_the_measure_verb(capsys) -> None:
    rc = main(["overview", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    verbs_section = next(s for s in payload["sections"] if s["title"] == "Verbs")
    assert any("measure" in item for item in verbs_section["items"])
