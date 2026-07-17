"""Tests for ``lobes.bench.compare`` — issue #81, task t9.

Cross-profile RUNTIME comparison: ``cortex-only`` / ``cortex+senses`` /
``senses-direct`` (straight :func:`lobes.roles_measure.measure_registry`
fan-out) and the catalog-gated ``qwen-nvfp4-vs-bf16`` profile.

No live model is ever required: "reachable" tests talk to a tiny canned local
HTTP server (the same recipe ``tests/test_cli_measure.py`` uses); "unreachable"
tests talk to a closed OS port (instant connection-refused).
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
import time

import pytest

from lobes import roles_measure as RM
from lobes.bench import compare as C
from lobes.catalog import SupportedModel
from lobes.roles import ROLES, RoleInfo

# ---------------------------------------------------------------------------
# Fixtures — a canned local server for the LLM (cortex/senses) wire shape.
# ---------------------------------------------------------------------------

_PROMPT_TOKENS = 100
_COMPLETION_TOKENS = 10
_SLEEP_S = 0.005


class _LLMHandler(http.server.BaseHTTPRequestHandler):
    """Canned responses for the chat-completions + health/metrics wire shapes."""

    def _write_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/metrics":
            body = b'vllm:gpu_cache_usage_perc{model_name="x"} 0.2\n'
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
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
                        "prompt_tokens": _PROMPT_TOKENS,
                        "completion_tokens": _COMPLETION_TOKENS,
                    },
                },
            )
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D102
        pass


@pytest.fixture()
def llm_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _LLMHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


def _closed_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _role_info(role: str, *, endpoint: str, loaded: bool = True, model: str = "m") -> RoleInfo:
    return RoleInfo(
        role=role,
        model=model,
        runtime="vllm",
        endpoint=endpoint,
        path="/v1/chat/completions",
        context=4096,
        quant="",
        mtp=False,
        tools=False,
        responsibilities=(),
        forbidden_responsibilities=(),
        ready=None,
        loaded=loaded,
    )


def _registry(*, cortex_endpoint: str, senses_endpoint: str = "", senses_loaded=False):
    reg = {r: _role_info(r, endpoint="", loaded=False) for r in ROLES}
    reg["cortex"] = _role_info("cortex", endpoint=cortex_endpoint, loaded=True)
    reg["senses"] = _role_info(
        "senses", endpoint=senses_endpoint, loaded=senses_loaded or bool(senses_endpoint)
    )
    return reg


_FAKE_NVFP4 = SupportedModel(
    id="test-org/Qwen3.6-27B-NVFP4-fake",
    role_hint="candidate",
    shape="dense",
    context="256K native",
    native_max_model_len=262144,
    tool_parser="qwen3_coder",
    quantization="modelopt_fp4",
    status="load-tested",
    doc="fake.md",
)

_FAKE_BF16 = SupportedModel(
    id="test-org/Qwen3.6-27B-BF16-fake",
    role_hint="candidate",
    shape="dense",
    context="256K native",
    native_max_model_len=262144,
    tool_parser="qwen3_coder",
    quantization="none",
    status="configured",
    doc="fake.md",
)

_FAKE_CATALOG_BOTH = (_FAKE_NVFP4, _FAKE_BF16)


# ---------------------------------------------------------------------------
# PROFILE_NAMES / role-mapping sanity
# ---------------------------------------------------------------------------


def test_profile_names_are_the_four_documented_profiles() -> None:
    assert C.PROFILE_NAMES == (
        "cortex-only",
        "cortex+senses",
        "senses-direct",
        "qwen-nvfp4-vs-bf16",
    )


# ---------------------------------------------------------------------------
# Acceptance 1 — cortex-only / cortex+senses emit COMPARABLE metric keys
# ---------------------------------------------------------------------------


def test_cortex_only_reachable(llm_server: str) -> None:
    registry = _registry(cortex_endpoint=llm_server)
    result = C.run_profile("cortex-only", registry, timeout=5.0)
    assert result["profile"] == "cortex-only"
    assert result["available"] is True
    assert result["reason"] is None
    assert set(result["columns"]) == {"cortex"}
    assert result["columns"]["cortex"]["ready"] is True
    assert set(result["columns"]["cortex"]["metrics"]) == set(RM.LLM_METRIC_KEYS)


def test_cortex_plus_senses_reachable_shares_metric_keys(llm_server: str) -> None:
    registry = _registry(cortex_endpoint=llm_server, senses_endpoint=llm_server)
    result = C.run_profile("cortex+senses", registry, timeout=5.0)
    assert result["available"] is True
    assert set(result["columns"]) == {"cortex", "senses"}
    cortex_keys = set(result["columns"]["cortex"]["metrics"])
    senses_keys = set(result["columns"]["senses"]["metrics"])
    # Comparable: the SAME metric-key vocabulary across profile columns.
    assert cortex_keys == senses_keys == set(RM.LLM_METRIC_KEYS)


def test_senses_direct_reachable(llm_server: str) -> None:
    registry = _registry(cortex_endpoint=llm_server, senses_endpoint=llm_server)
    result = C.run_profile("senses-direct", registry, timeout=5.0)
    assert result["available"] is True
    assert set(result["columns"]) == {"senses"}
    assert result["columns"]["senses"]["ready"] is True


# ---------------------------------------------------------------------------
# Acceptance 4 — graceful offline degradation: unreachable -> unavailable, no raise
# ---------------------------------------------------------------------------


def test_cortex_only_unreachable_marks_unavailable_not_exception() -> None:
    port = _closed_port()
    registry = _registry(cortex_endpoint=f"http://127.0.0.1:{port}")
    result = C.run_profile("cortex-only", registry, timeout=2.0)
    assert result["available"] is False
    assert result["reason"]
    assert result["columns"]["cortex"]["ready"] is False
    metrics = result["columns"]["cortex"]["metrics"]
    assert all(v is None for k, v in metrics.items() if k != "context")


def test_cortex_plus_senses_partially_unreachable_marks_unavailable() -> None:
    """cortex reachable, senses unreachable -> whole profile unavailable, no raise."""
    port = _closed_port()
    registry = _registry(
        cortex_endpoint=f"http://127.0.0.1:{port}", senses_endpoint="", senses_loaded=True
    )
    result = C.run_profile("cortex+senses", registry, timeout=2.0)
    assert result["available"] is False
    assert result["reason"]
    # Never raises, and still carries whatever data it could gather (both null here).
    assert set(result["columns"]) == {"cortex", "senses"}


# ---------------------------------------------------------------------------
# Acceptance 2 — qwen-nvfp4-vs-bf16: catalog-gated, both branches
# ---------------------------------------------------------------------------


def test_qwen_variants_missing_bf16_in_the_real_catalog_today() -> None:
    """Characterizes today's catalog: two NVFP4 27B Qwen entries, no bf16 27B."""
    nvfp4, bf16 = C.qwen_nvfp4_bf16_variants()
    assert nvfp4 is not None
    assert bf16 is None


def test_qwen_variants_both_present_with_injected_catalog() -> None:
    nvfp4, bf16 = C.qwen_nvfp4_bf16_variants(_FAKE_CATALOG_BOTH)
    assert nvfp4 is _FAKE_NVFP4
    assert bf16 is _FAKE_BF16


def test_qwen_profile_unavailable_when_catalog_missing_bf16_and_never_touches_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not probe the network when the catalog condition isn't met")

    monkeypatch.setattr(C, "measure_role", boom)
    registry = _registry(cortex_endpoint="http://127.0.0.1:1")  # would refuse instantly anyway
    result = C.run_profile("qwen-nvfp4-vs-bf16", registry, timeout=2.0)
    assert result["profile"] == "qwen-nvfp4-vs-bf16"
    assert result["available"] is False
    assert "catalog" in result["reason"].lower()
    assert result["columns"] == {}


def test_qwen_profile_available_when_catalog_has_both_and_endpoint_reachable(
    llm_server: str,
) -> None:
    registry = _registry(cortex_endpoint=llm_server)
    result = C.run_profile("qwen-nvfp4-vs-bf16", registry, timeout=5.0, catalog=_FAKE_CATALOG_BOTH)
    assert result["available"] is True
    assert result["reason"] is None
    assert set(result["columns"]) == {"nvfp4", "bf16"}
    assert result["columns"]["nvfp4"]["model"] == _FAKE_NVFP4.id
    assert result["columns"]["bf16"]["model"] == _FAKE_BF16.id
    for label in ("nvfp4", "bf16"):
        assert result["columns"][label]["ready"] is True
        assert set(result["columns"][label]["metrics"]) == set(RM.LLM_METRIC_KEYS)


def test_qwen_profile_unavailable_when_catalog_has_both_but_endpoint_unreachable() -> None:
    port = _closed_port()
    registry = _registry(cortex_endpoint=f"http://127.0.0.1:{port}")
    result = C.run_profile("qwen-nvfp4-vs-bf16", registry, timeout=2.0, catalog=_FAKE_CATALOG_BOTH)
    assert result["available"] is False
    assert result["reason"]
    assert set(result["columns"]) == {"nvfp4", "bf16"}
    for label in ("nvfp4", "bf16"):
        assert result["columns"][label]["ready"] is False


def test_qwen_profile_no_cortex_endpoint_degrades_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cortex unwired (endpoint='') -> loaded=False synthetic probe -> no network touched."""

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("must not probe the network when cortex has no endpoint")

    monkeypatch.setattr(RM._metrics, "probe_backend", boom)
    monkeypatch.setattr(RM._assess, "measure_prefill_ttft", boom)
    monkeypatch.setattr(RM._assess, "_post", boom)
    registry = {r: _role_info(r, endpoint="", loaded=False) for r in ROLES}
    result = C.run_profile("qwen-nvfp4-vs-bf16", registry, timeout=2.0, catalog=_FAKE_CATALOG_BOTH)
    assert result["available"] is False
    for label in ("nvfp4", "bf16"):
        assert result["columns"][label]["ready"] is False


# ---------------------------------------------------------------------------
# Acceptance 3 — RUNTIME-ONLY vocabulary across every profile
# ---------------------------------------------------------------------------


def test_every_profile_metric_key_is_runtime_only(llm_server: str) -> None:
    registry = _registry(cortex_endpoint=llm_server, senses_endpoint=llm_server)
    results = C.run_profiles(None, registry, timeout=5.0, catalog=_FAKE_CATALOG_BOTH)
    assert set(results) == set(C.PROFILE_NAMES)
    for profile_result in results.values():
        for column in profile_result["columns"].values():
            assert set(column["metrics"]) <= RM.ALLOWED_METRIC_KEYS


# ---------------------------------------------------------------------------
# run_profile / run_profiles plumbing
# ---------------------------------------------------------------------------


def test_run_profile_unknown_name_raises_value_error() -> None:
    registry = _registry(cortex_endpoint="http://127.0.0.1:1")
    with pytest.raises(ValueError):
        C.run_profile("not-a-real-profile", registry)


def test_run_profiles_defaults_to_all_four() -> None:
    registry = _registry(cortex_endpoint=f"http://127.0.0.1:{_closed_port()}")
    results = C.run_profiles(None, registry, timeout=1.0)
    assert set(results) == set(C.PROFILE_NAMES)


def test_run_profiles_selects_a_subset() -> None:
    registry = _registry(cortex_endpoint=f"http://127.0.0.1:{_closed_port()}")
    results = C.run_profiles(["cortex-only", "senses-direct"], registry, timeout=1.0)
    assert set(results) == {"cortex-only", "senses-direct"}
