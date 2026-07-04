"""Gateway ``GET /capabilities`` tests (issue #81, task t6).

The gateway surfaces the SIX first-class roles (cortex/senses/embedder/
reranker/stt/tts) as a read-only GET, built from the ONE canonical registry
builder in :mod:`lobes.roles` — the same one the CLI's ``capabilities --json``
(t5) uses — so the two payloads share exactly one shape: a dict keyed by role,
each value the full ``RoleInfo`` field set (``dataclasses.asdict``).

``ready``: the pure ``capabilities_payload`` builder derives it from ``loaded``
by default (``audio_ready`` unset), so the CLI/unit path keeps a present
boolean without opening a socket. The HTTP route (``do_GET`` /capabilities)
now injects a *live* stt/tts readiness probe (``probe_audio_ready``, issue #89)
and a client-reachable origin (``reachable_origin``, issue #87) — exercised by
the loopback tests below.
"""

from __future__ import annotations

import dataclasses
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.roles import ROLES, build_role_registry

_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_MULTIMODAL_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"


def _full_env(**over) -> dict[str, str]:
    """A fully-wired six-role generate+pooling fleet (audio deliberately unset)."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _PRIMARY_ID,
        "PRIMARY_MAX_MODEL_LEN": "131072",
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _MULTIMODAL_ID,
        "MULTIMODAL_MAX_MODEL_LEN": "32768",
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
    }
    env.update(over)
    return env


# --- capabilities_payload: pure, offline (no sockets) -----------------------


def test_capabilities_payload_matches_cli_shape() -> None:
    # The CLI's `capabilities --json` emits {role: dataclasses.asdict(RoleInfo)};
    # the gateway payload must be byte-for-byte the same shape, modulo `ready`
    # (the CLI leaves it None too; the gateway fills it from `loaded`).
    env = _full_env()
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env)
    assert set(payload) == set(ROLES)
    registry = build_role_registry(table, cfg, env=env)
    for role in ROLES:
        expected = dataclasses.asdict(registry[role])
        expected["ready"] = registry[role].loaded
        assert payload[role] == expected


def test_capabilities_payload_required_keys_present_for_every_role() -> None:
    env = _full_env()
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env)
    for role in ROLES:
        entry = payload[role]
        for key in ("endpoint", "model", "context", "ready", "responsibilities"):
            assert key in entry
        assert isinstance(entry["ready"], bool)  # never None on the gateway response


def test_capabilities_payload_reports_served_context_not_a_literal() -> None:
    # cortex/senses report the deployment's SERVED --max-model-len (from env),
    # proving the handler reads the registry/config rather than a hardcoded value.
    env = _full_env()
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env)
    assert payload["cortex"]["context"] == 131072
    assert payload["senses"]["context"] == 32768
    assert payload["cortex"]["model"] == _PRIMARY_ID
    assert payload["senses"]["model"] == _MULTIMODAL_ID


def test_capabilities_payload_no_hardcoded_model_id() -> None:
    # Renaming the operator's served model changes what cortex reports — proves
    # the handler never bakes in a literal model id.
    env = _full_env(PRIMARY_SERVED_NAME="acme/custom-27b")
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env)
    assert payload["cortex"]["model"] == "acme/custom-27b"


def test_capabilities_payload_unwired_roles_present_not_omitted() -> None:
    # A primary-only fleet: no multimodal/embed/rerank/audio wired.
    env = {"PRIMARY_URL": "http://vllm-primary:8000", "PRIMARY_SERVED_NAME": _PRIMARY_ID}
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env)
    assert set(payload) == set(ROLES)  # all six present, never a 500, never omitted
    for role in ("senses", "embedder", "reranker", "stt", "tts"):
        assert payload[role]["loaded"] is False
        assert payload[role]["ready"] is False  # falsey — not wired, no live probe (t8)
    assert payload["cortex"]["loaded"] is True
    assert payload["cortex"]["ready"] is True


def test_capabilities_payload_ready_mirrors_loaded() -> None:
    env = _full_env()
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env)
    for role in ROLES:
        assert payload[role]["ready"] == payload[role]["loaded"]


def test_capabilities_payload_defaults_env_to_os_environ(monkeypatch) -> None:
    # Inside the gateway container, os.environ IS the deployment env — the `env`
    # kwarg must default to it, not require a caller to pass one explicitly.
    monkeypatch.setenv("PRIMARY_URL", "http://vllm-primary:8000")
    monkeypatch.setenv("PRIMARY_SERVED_NAME", _PRIMARY_ID)
    monkeypatch.setenv("PRIMARY_MAX_MODEL_LEN", "131072")
    table, cfg = build_config()
    payload = S.capabilities_payload(table, cfg)
    assert payload["cortex"]["context"] == 131072


# --- /capabilities is advertised -------------------------------------------


def test_capabilities_endpoint_is_advertised() -> None:
    table, _cfg = build_config(_full_env())
    assert "GET /capabilities" in S._endpoints_for(table, False)
    assert "GET /capabilities" in S._endpoints_for(table, True)


# --- loopback: the real HTTP route ------------------------------------------


@pytest.fixture
def capabilities_gateway(monkeypatch):
    """A real ThreadingHTTPServer wired with a full six-role env (audio unset)."""
    env = _full_env()
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    table, cfg = build_config(env)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_integration_get_capabilities(capabilities_gateway) -> None:
    with urllib.request.urlopen(capabilities_gateway + "/capabilities", timeout=5) as r:
        assert r.status == 200
        assert r.headers.get_content_type() == "application/json"
        payload = json.load(r)
    assert set(payload) == set(ROLES)
    for role in ROLES:
        entry = payload[role]
        for key in ("endpoint", "model", "context", "ready", "responsibilities"):
            assert key in entry
    assert payload["cortex"]["context"] == 131072
    assert payload["senses"]["context"] == 32768
    # Both gateway-fronted roles resolve to the same caller-facing base URL.
    assert payload["cortex"]["endpoint"] == payload["senses"]["endpoint"] != ""
    # Not wired in this fixture (no AUDIO_URL) — present, unloaded, not a 500.
    assert payload["stt"]["loaded"] is False
    assert payload["stt"]["ready"] is False


def test_integration_get_capabilities_is_idempotent_read_only(capabilities_gateway) -> None:
    with urllib.request.urlopen(capabilities_gateway + "/capabilities", timeout=5) as r1:
        first = json.load(r1)
    with urllib.request.urlopen(capabilities_gateway + "/capabilities", timeout=5) as r2:
        second = json.load(r2)
    assert first == second  # a GET never mutates fleet/routing state


# --- #87: the reachable origin (request Host header + GATEWAY_PUBLIC_URL) ----


def test_reachable_origin_prefers_public_url_then_host() -> None:
    # Explicit GATEWAY_PUBLIC_URL wins (a tunnel / Host-rewriting proxy)...
    assert (
        S.reachable_origin("localhost:8001", "https://tunnel.example/") == "https://tunnel.example"
    )
    # ...else echo the origin the client actually dialed (the Host header)...
    assert S.reachable_origin("gw.example:8001", None) == "http://gw.example:8001"
    # ...else None → the caller falls back to the config-derived origin.
    assert S.reachable_origin(None, None) is None


def test_build_config_reads_gateway_public_url() -> None:
    _, cfg = build_config(_full_env(GATEWAY_PUBLIC_URL="https://tunnel.example/"))
    assert cfg.public_url == "https://tunnel.example"  # trailing slash trimmed
    _, cfg2 = build_config(_full_env())
    assert cfg2.public_url is None  # unset → None (Host-header fallback)


def test_capabilities_payload_gateway_url_applies_to_all_roles() -> None:
    env = _full_env(AUDIO_URL="http://realtime:8080")
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url="https://tunnel.example")
    for role in ROLES:
        assert payload[role]["endpoint"] == "https://tunnel.example"


def test_capabilities_payload_threads_audio_ready() -> None:
    env = _full_env(AUDIO_URL="http://realtime:8080")
    table, cfg = build_config(env)
    # audio_ready=False → stt/tts advertise ready:false even though AUDIO_URL is set.
    warming = S.capabilities_payload(table, cfg, env=env, audio_ready=False)
    for role in ("stt", "tts"):
        assert warming[role]["ready"] is False
    # audio_ready=True → ready.
    live = S.capabilities_payload(table, cfg, env=env, audio_ready=True)
    for role in ("stt", "tts"):
        assert live[role]["ready"] is True


def test_integration_capabilities_endpoint_reflects_request_host(capabilities_gateway) -> None:
    """#87: every gateway-fronted role's endpoint is the origin the client dialed."""
    import http.client

    host_port = capabilities_gateway.removeprefix("http://")
    conn = http.client.HTTPConnection(host_port, timeout=5)
    conn.putrequest("GET", "/capabilities", skip_host=True)
    conn.putheader("Host", "gw.example:8001")
    conn.endheaders()
    resp = conn.getresponse()
    payload = json.load(resp)
    conn.close()
    for role in ("cortex", "senses", "embedder", "reranker"):
        assert payload[role]["endpoint"] == "http://gw.example:8001"
