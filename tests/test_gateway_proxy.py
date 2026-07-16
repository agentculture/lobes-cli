"""The proxy data plane — follow the referral (proxy-lobes t6, issues #115/#127).

The THIRD lobe state: awake (hosted) / asleep (referral-only 404) / **PROXY** —
a dropped role whose operator armed ``<PREFIX>_PEER_PROXY`` is answered by
FORWARDING the request to the operator-declared peer origin, instead of the
referral 404. Everything here drives :func:`lobes.gateway.server.handle_post`'s
pure seam (an injected ``open_upstream``, no sockets) plus a loopback
integration at the end, mirroring tests/test_gateway_server.py conventions.

Contract under test (the plan's acceptance criteria, verbatim):

(a) a dropped+proxied role's request forwards to the peer origin (original
    path, model rewritten to the peer's served id) and the peer's JSON/SSE
    answer relays back unchanged;
(b) the outbound request carries the per-peer key and NEVER the caller's own
    Authorization — captured outbound headers are grepped for the caller's
    token and must contain nothing;
(c) a request already carrying the hop marker that would depart again is
    refused (``proxy_loop``) with zero outbound attempts — single hop only;
(d) peer connect-refused/timeout ⇒ retryable 503 + Retry-After (the existing
    owner-down conventions, #14/#91 — never a cross-model fallback), and the
    LOCAL pressure policy is never applied to a proxied request (the peer's
    own gateway applies its policy);
(e) the peer answering 404 ``role_infeasible`` (a misdeclared referral) is
    TERMINAL: the relayed error names the peer, and no second hop is ever
    attempted;
(f) every proxied response carries ``X-Lobes-Proxied-By: <peer origin>``
    verbatim; locally-served responses never do.

Plus the advertisement half: ``/v1/models`` lists a proxied role's served id
IFF the live peer-readiness signal is ``True`` (peer down ⇒ the id drops,
exactly like a dead local backend), and ``GET /capabilities`` threads the peer
signal into the proxied role's ``ready`` (honesty h2: a live proxied-path
probe or honestly not-ready — never hardcoded true).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._routing import list_models_payload
from lobes.roles import build_role_registry

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"  # the catalog multimodal default
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
_GATEWAY_URL = "http://localhost:8000"

_THOR_ORIGIN = "http://thor.local:8001"
_SPARK_ORIGIN = "http://spark.local:8001"

_PEER_KEY = "sk-peer-outbound-pairwise-0001"
_CALLER_TOKEN = "sk-caller-inbound-token-9999"

# A swap/iowait sample the pressure policy sheds cortex/senses under.
_HIGH_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 90.0}
_NO_PRESSURE = {"swap_used_percent": 0.0, "iowait_percent": 0.0}


# --- env simulators (mirror tests/test_dropped_lobe_honesty.py) --------------


def _spark_env(**over) -> dict[str, str]:
    """spark-lobe + proxy: cortex/pooling hosted, senses DROPPED (unwired) and
    proxied to the Thor peer with a pairwise outbound key."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
        "MULTIMODAL_FEASIBLE": "false",
        "MULTIMODAL_PEER_ORIGIN": _THOR_ORIGIN,
        "MULTIMODAL_PEER_PROXY": "true",
        "MULTIMODAL_PEER_API_KEY": _PEER_KEY,
    }
    env.update(over)
    return env


def _thor_env(**over) -> dict[str, str]:
    """thor-lobe + proxy: senses/pooling hosted, cortex DROPPED (the primary is
    unconditionally WIRED but infeasible) and proxied to the Spark peer."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "PRIMARY_FEASIBLE": "false",
        "PRIMARY_PEER_ORIGIN": _SPARK_ORIGIN,
        "PRIMARY_PEER_PROXY": "true",
        "PRIMARY_PEER_API_KEY": _PEER_KEY,
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _SENSES_ID,
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
    }
    env.update(over)
    return env


def _build(env):
    table, cfg = build_config(env)
    return table, cfg, S.peer_specs_from_table(table, env)


# --- fakes (mirror tests/test_gateway_server.py) ------------------------------


class _FakeUpstream:
    """Duck-typed stand-in for server._Upstream (no socket)."""

    def __init__(self, status, body=b'{"ok":1}', chunks=None, headers=None):
        self.status = status
        self.headers = headers if headers is not None else [("Content-Type", "application/json")]
        self._body = body
        self._chunks = list(chunks) if chunks is not None else None
        self.closed = False

    def read_all(self):
        return self._body

    def read(self, _n):
        if self._chunks is None:
            data, self._body = self._body, b""
            return data
        return self._chunks.pop(0) if self._chunks else b""

    def close(self):
        self.closed = True


def _opener(outcome=200, body=b'{"ok":1}', chunks=None):
    """An ``open_upstream`` stub recording every dial (backend, path, body,
    headers). ``outcome`` may be an int status or an Exception to raise."""
    calls = []

    def opener(backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        calls.append(
            SimpleNamespace(backend=backend, path=path, body=fwd_body, headers=list(headers))
        )
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeUpstream(outcome, body=body, chunks=chunks)

    return opener, calls


def _post(table, cfg, specs, body, *, headers=(), pressure=None, opener=None, calls=None):
    if opener is None:
        opener, calls = _opener()
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        list(headers),
        body,
        opener,
        pressure=pressure,
        peer_specs=specs,
    )
    return resp, calls


# ============================================================================
# peer_specs_from_table — the served-id source-of-truth resolution
# ============================================================================


def test_peer_specs_unwired_role_served_name_from_env() -> None:
    # The dropped role's container is absent (no Backend in the table), but the
    # deployment env still declares what the role serves: <PREFIX>_SERVED_NAME
    # is the honest source for the id this box forwards/advertises.
    env = _spark_env(MULTIMODAL_SERVED_NAME="custom/gemma-tuned")
    table, _cfg, specs = _build(env)
    assert set(specs) == {"multimodal"}
    spec = specs["multimodal"]
    assert spec.origin == _THOR_ORIGIN
    assert spec.served_name == "custom/gemma-tuned"
    assert spec.api_key == _PEER_KEY


def test_peer_specs_unwired_role_served_name_falls_back_to_catalog() -> None:
    # No <PREFIX>_SERVED_NAME either → the catalog canonical id for the role
    # (the same source lobes.roles uses to NAME an unwired role's model).
    table, _cfg, specs = _build(_spark_env())
    assert specs["multimodal"].served_name == _SENSES_ID


def test_peer_specs_wired_but_infeasible_role_uses_backend_served_name() -> None:
    # thor-lobe: the primary is unconditionally wired, so the WIRED backend's
    # served_name outranks env/catalog (it is what the table itself declares).
    table, _cfg, specs = _build(_thor_env())
    assert set(specs) == {"primary"}
    assert specs["primary"].served_name == _CORTEX_ID
    assert specs["primary"].origin == _SPARK_ORIGIN


def test_peer_specs_empty_without_proxy_config() -> None:
    # Referral-only (origin, no knob) and no-peer deployments build ZERO specs.
    env = _spark_env()
    del env["MULTIMODAL_PEER_PROXY"]
    table, _cfg, specs = _build(env)
    assert specs == {}
    table, _cfg = build_config({"PRIMARY_SERVED_NAME": _CORTEX_ID})
    assert S.peer_specs_from_table(table, {}) == {}


def test_peer_specs_key_never_in_repr() -> None:
    # PeerSpec.api_key is repr=False; the mapping's repr must not leak it either.
    _table, _cfg, specs = _build(_spark_env())
    assert _PEER_KEY not in repr(specs)
    assert _PEER_KEY not in str(specs)


# ============================================================================
# (a) forward: original path, rewritten model, response relayed unchanged
# ============================================================================


@pytest.mark.parametrize("alias", ["senses", "multimodal", "normal", _SENSES_ID])
def test_proxied_request_forwards_to_peer_origin(alias: str) -> None:
    # Every alias of the dropped+proxied role — role identity, capability tier,
    # back-compat synonym, AND the concrete served id (unwired, so the id is
    # not in the table — it resolves via the peer spec) — forwards.
    table, cfg, specs = _build(_spark_env())
    body = json.dumps({"model": alias, "messages": []}).encode()
    resp, calls = _post(table, cfg, specs, body)
    assert len(calls) == 1
    call = calls[0]
    assert call.backend.base_url == _THOR_ORIGIN  # the declared origin, verbatim
    assert call.path == "/v1/chat/completions"  # the original path
    assert json.loads(call.body)["model"] == _SENSES_ID  # rewritten to the served id
    assert resp.status == 200
    assert resp.upstream is not None  # relayed, not gateway-generated
    assert resp.upstream.read_all() == b'{"ok":1}'  # the peer's answer, unchanged


def test_proxied_response_carries_proxied_by_header_verbatim() -> None:
    table, cfg, specs = _build(_spark_env())
    resp, _calls = _post(table, cfg, specs, b'{"model":"senses"}')
    assert dict(resp.headers)[S.PROXIED_BY_HEADER] == _THOR_ORIGIN


def test_proxied_sse_stream_relays_unchanged() -> None:
    chunks = [b'data: {"delta":1}\n\n', b"data: [DONE]\n\n"]
    opener, calls = _opener(200, chunks=list(chunks))
    table, cfg, specs = _build(_spark_env())
    body = json.dumps({"model": "senses", "stream": True}).encode()
    resp, _ = _post(table, cfg, specs, body, opener=opener, calls=calls)
    assert resp.status == 200
    assert resp.streaming is True  # the handler will chunk-relay it
    assert dict(resp.headers)[S.PROXIED_BY_HEADER] == _THOR_ORIGIN
    relayed = []
    while True:
        chunk = resp.upstream.read(65536)
        if not chunk:
            break
        relayed.append(chunk)
    assert relayed == chunks  # frames relayed as-is, in order


def test_thor_proxied_cortex_aliases_forward() -> None:
    # The mirror shape: cortex dropped (wired-but-infeasible) on thor-lobe.
    table, cfg, specs = _build(_thor_env())
    for alias in ("cortex", "main", "hard", _CORTEX_ID):
        resp, calls = _post(table, cfg, specs, json.dumps({"model": alias}).encode())
        assert resp.status == 200, alias
        assert len(calls) == 1 and calls[0].backend.base_url == _SPARK_ORIGIN, alias
        assert json.loads(calls[0].body)["model"] == _CORTEX_ID, alias


def test_unspecified_model_routing_to_proxied_default_forwards() -> None:
    # An UNSPECIFIED model routes to default_model (thor-lobe: the dropped
    # cortex) — with the proxy armed that default now forwards instead of
    # 404ing role_infeasible.
    table, cfg, specs = _build(_thor_env())
    resp, calls = _post(table, cfg, specs, b"{}")
    assert resp.status == 200
    assert len(calls) == 1 and calls[0].backend.base_url == _SPARK_ORIGIN


# ============================================================================
# (b) outbound credentials: pairwise key in, caller's token provably OUT
# ============================================================================


def test_outbound_carries_peer_key_and_never_callers_token() -> None:
    table, cfg, specs = _build(_spark_env())
    inbound = [
        ("Authorization", f"Bearer {_CALLER_TOKEN}"),
        ("Content-Type", "application/json"),
        ("X-Custom", "kept"),
    ]
    _resp, calls = _post(table, cfg, specs, b'{"model":"senses"}', headers=inbound)
    headers = calls[0].headers
    auth_values = [v for k, v in headers if k.lower() == "authorization"]
    assert auth_values == [f"Bearer {_PEER_KEY}"]  # exactly one, the pairwise key
    # Grep EVERYTHING that left the box for the caller's token: nothing.
    dumped = json.dumps(headers) + calls[0].body.decode("utf-8", errors="replace")
    assert _CALLER_TOKEN not in dumped
    # Non-credential headers still forward.
    assert ("X-Custom", "kept") in headers


def test_outbound_has_no_authorization_when_no_pairwise_key() -> None:
    env = _spark_env()
    del env["MULTIMODAL_PEER_API_KEY"]
    table, cfg, specs = _build(env)
    inbound = [("Authorization", f"Bearer {_CALLER_TOKEN}")]
    _resp, calls = _post(table, cfg, specs, b'{"model":"senses"}', headers=inbound)
    auth_values = [v for k, v in calls[0].headers if k.lower() == "authorization"]
    assert auth_values == []  # no key declared → NO header, never the caller's


def test_outbound_carries_single_hop_marker() -> None:
    table, cfg, specs = _build(_spark_env())
    _resp, calls = _post(table, cfg, specs, b'{"model":"senses"}')
    marker = [v for k, v in calls[0].headers if k.lower() == S.PROXIED_HEADER.lower()]
    assert marker == ["multimodal"]  # the proxied role's backend name — no origin, no key


# ============================================================================
# (c) loop guard — single hop only
# ============================================================================


def test_marked_request_that_would_reproxy_is_refused_zero_outbound() -> None:
    table, cfg, specs = _build(_spark_env())
    inbound = [(S.PROXIED_HEADER, "primary")]  # already crossed one hop elsewhere
    resp, calls = _post(table, cfg, specs, b'{"model":"senses"}', headers=inbound)
    assert calls == []  # NO outbound attempt
    assert resp.status == 508
    err = json.loads(resp.body)["error"]
    assert err["type"] == "proxy_loop" and err["code"] == "proxy_loop"
    # Both hops named: the hop already taken (the arriving marker) and the hop
    # refused (this box's declared peer origin for the role).
    assert err["hops"] == ["primary", _THOR_ORIGIN]
    assert "primary" in err["message"] and _THOR_ORIGIN in err["message"]
    # Nothing was proxied → no proxied-by header, and no key material anywhere.
    assert S.PROXIED_BY_HEADER not in dict(resp.headers)
    assert _PEER_KEY not in resp.body.decode()


def test_marker_header_is_case_insensitive() -> None:
    table, cfg, specs = _build(_spark_env())
    resp, calls = _post(
        table, cfg, specs, b'{"model":"senses"}', headers=[("x-lobes-proxied", "primary")]
    )
    assert resp.status == 508 and calls == []


def test_marked_request_for_locally_served_role_processes_normally() -> None:
    # An arriving marked request whose role IS hosted here never touches the
    # proxy branch: it forwards to the local backend exactly as before.
    table, cfg, specs = _build(_spark_env())
    inbound = [(S.PROXIED_HEADER, "multimodal")]
    resp, calls = _post(table, cfg, specs, b'{"model":"cortex"}', headers=inbound)
    assert resp.status == 200
    assert len(calls) == 1 and calls[0].backend.name == "primary"  # the LOCAL owner
    assert S.PROXIED_BY_HEADER not in dict(resp.headers)  # served locally


# ============================================================================
# (d) peer down ⇒ retryable 503; local pressure never applies to proxied
# ============================================================================


def test_peer_connect_refused_yields_503_with_retry_after() -> None:
    opener, calls = _opener(S.UpstreamError("peer:multimodal: connection refused"))
    table, cfg, specs = _build(_spark_env())
    resp, _ = _post(table, cfg, specs, b'{"model":"senses"}', opener=opener, calls=calls)
    assert len(calls) == 1  # tried the peer once — no second hop, no fallback (#91)
    assert resp.status == 503 and resp.upstream is None
    headers = dict(resp.headers)
    assert headers["Retry-After"] == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)
    assert headers[S.PROXIED_BY_HEADER] == _THOR_ORIGIN  # names the peer that failed
    err = json.loads(resp.body)["error"]
    assert err["type"] == "backend_unavailable"  # the existing owner-down convention
    assert _THOR_ORIGIN in err["message"]
    assert _PEER_KEY not in resp.body.decode()


def test_peer_5xx_yields_503_backend_unavailable() -> None:
    opener, calls = _opener(502)
    table, cfg, specs = _build(_spark_env())
    resp, _ = _post(table, cfg, specs, b'{"model":"senses"}', opener=opener, calls=calls)
    assert len(calls) == 1
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["type"] == "backend_unavailable"
    assert dict(resp.headers)["Retry-After"] == str(S.BACKEND_UNAVAILABLE_RETRY_AFTER_SECONDS)


def test_proxied_request_bypasses_local_pressure_shed() -> None:
    # Local pressure would shed a senses/multimodal tier request with 429 —
    # but a PROXIED one must forward regardless: the peer's own gateway applies
    # its own pressure policy on arrival, and shedding here too would
    # double-gate the role on the WRONG box's load.
    table, cfg, specs = _build(_spark_env())
    resp, calls = _post(table, cfg, specs, b'{"model":"senses"}', pressure=_HIGH_PRESSURE)
    assert resp.status == 200  # forwarded, not shed
    assert len(calls) == 1 and calls[0].backend.base_url == _THOR_ORIGIN
    # Control: a HOSTED full tier under the same pressure is still shed (429) —
    # the bypass is scoped to proxied names only.
    resp, calls = _post(table, cfg, specs, b'{"model":"main"}', pressure=_HIGH_PRESSURE)
    assert resp.status == 429 and calls == []


# ============================================================================
# (e) peer declines with role_infeasible ⇒ terminal, names the peer, one hop
# ============================================================================


def test_peer_role_infeasible_is_terminal_and_names_the_peer() -> None:
    peer_404 = json.dumps(
        {
            "error": {
                "message": "The model `senses` is not feasible on this machine — ...",
                "type": "role_infeasible",
                "code": "role_infeasible",
            }
        }
    ).encode()
    opener, calls = _opener(404, body=peer_404)
    table, cfg, specs = _build(_spark_env())
    resp, _ = _post(table, cfg, specs, b'{"model":"senses"}', opener=opener, calls=calls)
    assert len(calls) == 1  # never a second attempt / another hop
    assert resp.status == 404 and resp.upstream is None
    assert dict(resp.headers)[S.PROXIED_BY_HEADER] == _THOR_ORIGIN
    err = json.loads(resp.body)["error"]
    assert err["type"] == "role_infeasible" and err["code"] == "role_infeasible"
    # The message makes clear the PEER declined (a misdeclared referral).
    assert _THOR_ORIGIN in err["message"]
    assert "declined" in err["message"]
    assert _PEER_KEY not in resp.body.decode()


def test_peer_plain_404_is_relayed_verbatim() -> None:
    # A peer 404 that is NOT role_infeasible (e.g. its own model_not_found) is
    # the owner's authoritative client-error verdict — relayed as-is (#91).
    peer_404 = json.dumps(
        {"error": {"message": "The model `x` does not exist.", "type": "model_not_found"}}
    ).encode()
    opener, calls = _opener(404, body=peer_404)
    table, cfg, specs = _build(_spark_env())
    resp, _ = _post(table, cfg, specs, b'{"model":"senses"}', opener=opener, calls=calls)
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "model_not_found"  # untouched
    assert dict(resp.headers)[S.PROXIED_BY_HEADER] == _THOR_ORIGIN


def test_peer_other_4xx_relayed_like_single_owner_rules() -> None:
    # e.g. the peer's own pressure shed (429) rides back to the caller verbatim.
    opener, calls = _opener(429, body=b'{"error":{"type":"server_busy"}}')
    table, cfg, specs = _build(_spark_env())
    resp, _ = _post(table, cfg, specs, b'{"model":"senses"}', opener=opener, calls=calls)
    assert resp.status == 429
    assert dict(resp.headers)[S.PROXIED_BY_HEADER] == _THOR_ORIGIN


# ============================================================================
# precedence: non-proxied paths are byte-identical
# ============================================================================


def test_non_proxied_infeasible_role_keeps_byte_identical_referral_404() -> None:
    # Referral-only (origin declared, knob NOT armed): the exact pre-proxy 404,
    # byte for byte — proven against the same request with no peer_specs wired.
    env = _spark_env()
    del env["MULTIMODAL_PEER_PROXY"]
    table, cfg = build_config(env)
    specs = S.peer_specs_from_table(table, env)
    assert specs == {}  # sanity: nothing is proxied
    for alias in ("senses", "multimodal", "normal"):
        body = json.dumps({"model": alias}).encode()
        with_specs, calls = _post(table, cfg, specs, body)
        opener, legacy_calls = _opener()
        legacy = S.handle_post(table, cfg, "/v1/chat/completions", [], body, opener, pressure=None)
        assert calls == [] and legacy_calls == []
        assert with_specs.status == legacy.status == 404
        assert with_specs.body == legacy.body  # byte-identical referral body
        assert with_specs.headers == legacy.headers
        assert json.loads(with_specs.body)["error"]["code"] == "role_infeasible"
        assert json.loads(with_specs.body)["error"]["hosted_by"] == _THOR_ORIGIN


def test_unknown_model_still_404s_model_not_found_never_proxied() -> None:
    # h23 precedence survives the proxy branch: an id that was never advertised
    # anywhere (not wired, not an alias, not the proxied role's served id) is a
    # model_not_found 404 — never silently forwarded to the peer under the
    # default model's identity.
    table, cfg, specs = _build(_thor_env())  # default_model routes to the PROXIED cortex
    resp, calls = _post(table, cfg, specs, b'{"model":"never-advertised-id"}')
    assert calls == []
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["code"] == "model_not_found"


def test_locally_served_response_never_carries_proxied_by() -> None:
    table, cfg, specs = _build(_spark_env())
    for model in ("cortex", "main", _CORTEX_ID, _EMBED_ID):
        resp, calls = _post(table, cfg, specs, json.dumps({"model": model}).encode())
        assert resp.status == 200, model
        assert S.PROXIED_BY_HEADER not in dict(resp.headers), model
        assert len(calls) == 1 and calls[0].backend.base_url != _THOR_ORIGIN, model


def test_handle_post_without_peer_specs_is_pre_proxy_behaviour() -> None:
    # A caller that never passes peer_specs (every pre-t6 call site, and any
    # deployment with no proxy config) gets the referral 404 even for a name
    # the TABLE marks proxied — the data plane only exists once specs are wired.
    table, cfg = build_config(_spark_env())
    assert "multimodal" in table.peer_proxied  # the table itself is armed
    opener, calls = _opener()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"senses"}', opener)
    assert resp.status == 404 and calls == []
    assert json.loads(resp.body)["error"]["code"] == "role_infeasible"


# ============================================================================
# /v1/models — advertised IFF the peer-readiness signal is True
# ============================================================================


def _ready(**over):
    ready = {"primary": True, "embed": True, "rerank": True}
    ready.update(over)
    return ready


def test_v1_models_lists_proxied_id_iff_peer_ready_true() -> None:
    table, _cfg, specs = _build(_spark_env())
    peer_served = {name: spec.served_name for name, spec in specs.items()}
    # Peer verified up (the probe checked its /v1/models lists the id) → listed.
    ids = {
        m["id"] for m in list_models_payload(table, _ready(multimodal=True), peer_served)["data"]
    }
    assert _SENSES_ID in ids and _CORTEX_ID in ids
    # Peer down / unprobed / missing → dropped, exactly like a dead local backend.
    for signal in (False, None):
        ids = {
            m["id"]
            for m in list_models_payload(table, _ready(multimodal=signal), peer_served)["data"]
        }
        assert _SENSES_ID not in ids, signal
    ids = {m["id"] for m in list_models_payload(table, _ready(), peer_served)["data"]}
    assert _SENSES_ID not in ids  # no signal at all


def test_v1_models_no_ready_snapshot_never_lists_proxied_id() -> None:
    # ready=None (no live cache — the offline path) lists every wired local
    # backend, but a proxied id needs an affirmative LIVE peer signal: without
    # one it is never advertised (h2 — no hardcoded reachability claims).
    table, _cfg, specs = _build(_spark_env())
    peer_served = {name: spec.served_name for name, spec in specs.items()}
    payload = list_models_payload(table, None, peer_served)
    ids = {m["id"] for m in payload["data"]}
    assert _SENSES_ID not in ids
    assert _CORTEX_ID in ids


def test_v1_models_ignores_peer_served_for_non_proxied_names() -> None:
    # Belt and braces: peer_served entries for names NOT in table.peer_proxied
    # are never listed — the routing table's opt-in is the only gate.
    env = _spark_env()
    del env["MULTIMODAL_PEER_PROXY"]  # referral-only now
    table, _cfg = build_config(env)
    payload = list_models_payload(table, _ready(multimodal=True), {"multimodal": _SENSES_ID})
    assert _SENSES_ID not in {m["id"] for m in payload["data"]}


def test_v1_models_without_peer_served_is_unchanged() -> None:
    table, _cfg, _specs = _build(_spark_env())
    baseline = list_models_payload(table, _ready())
    assert baseline == list_models_payload(table, _ready(), None)
    assert {m["id"] for m in baseline["data"]} == {_CORTEX_ID, _EMBED_ID, _RERANK_ID}


# ============================================================================
# GET /capabilities — proxied ready follows the live peer signal (h2)
# ============================================================================


def test_capabilities_proxied_ready_follows_peer_signal() -> None:
    env = _spark_env()
    table, cfg = build_config(env)
    for signal, expected in ((True, True), (False, False), (None, False)):
        payload = S.capabilities_payload(
            table,
            cfg,
            env=env,
            gateway_url=_GATEWAY_URL,
            backend_ready=_ready(multimodal=signal),
        )
        senses = payload["senses"]
        assert senses["ready"] is expected, signal
        assert senses["proxied"] is True
        assert senses["hosted_by"] == _THOR_ORIGIN
        assert senses["feasible"] is False  # still a hardware fact, never relaxed


def test_capabilities_proxied_ready_false_without_live_signal() -> None:
    # No backend_ready at all (the offline/CLI path): honestly not-ready,
    # never hardcoded true (h2).
    env = _spark_env()
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert payload["senses"]["ready"] is False
    assert payload["senses"]["proxied"] is True


def test_capabilities_referral_only_ready_stays_clamped_false() -> None:
    # The non-proxied path is byte-identical: a referral-only dropped role's
    # ready stays clamped False even with a stray live True signal (the t5 pin).
    env = _spark_env()
    del env["MULTIMODAL_PEER_PROXY"]
    table, cfg = build_config(env)
    payload = S.capabilities_payload(
        table, cfg, env=env, gateway_url=_GATEWAY_URL, backend_ready=_ready(multimodal=True)
    )
    assert payload["senses"]["ready"] is False
    assert "proxied" not in payload["senses"]


def test_build_role_registry_peer_ready_channel_is_scoped_to_proxied_names() -> None:
    # The NEW peer_ready channel (the live proxied-path probe) flips ready only
    # for a PROXIED role; backend_ready (the LOCAL probe channel) still never
    # does — the two signals stay distinct, as t5's clamp docstring demanded.
    table, cfg = build_config(_spark_env())
    reg = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL, peer_ready={"multimodal": True})
    assert reg["senses"].ready is True
    assert reg["senses"].feasible is False  # unchanged hardware fact
    # peer_ready never resurrects a non-proxied role.
    env = _spark_env()
    del env["MULTIMODAL_PEER_PROXY"]
    table, cfg = build_config(env)
    reg = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL, peer_ready={"multimodal": True})
    assert reg["senses"].ready is False


# ============================================================================
# wiring: serve() builds PeerSpecs and hands them to the ReadinessCache
# ============================================================================


def test_serve_wires_peer_specs_into_cache_and_handler(monkeypatch) -> None:
    env = _spark_env()
    table, cfg = build_config(env)
    expected_specs = S.peer_specs_from_table(table)  # serve() reads os.environ

    recorded: dict = {}

    def fake_from_backends(backends, **kwargs):
        recorded["backends"] = list(backends)
        recorded.update(kwargs)
        return SimpleNamespace(refresh=lambda: None, start=lambda: None, current=lambda: {})

    class _StubServer:
        def __init__(self, addr, handler):
            recorded["handler"] = handler

        def serve_forever(self):
            raise SystemExit  # stop serve() after wiring

    monkeypatch.setattr(S.ReadinessCache, "from_backends", fake_from_backends)
    monkeypatch.setattr(S, "ThreadingHTTPServer", _StubServer)
    with pytest.raises(SystemExit):
        S.serve(table, cfg)
    # The cache got one PeerSpec per proxied role, from the routing table.
    assert tuple(recorded["peer_specs"]) == tuple(expected_specs.values())
    # The handler carries the same specs for the data plane + /v1/models.
    assert recorded["handler"].peer_specs == expected_specs


# ============================================================================
# loopback integration: the full handler path over HTTP
# ============================================================================


@pytest.fixture
def proxy_gateway(monkeypatch):
    env = _spark_env()
    table, cfg = build_config(env)
    specs = S.peer_specs_from_table(table, env)
    opened: list = []

    def fake_open(backend, path, body, headers, *, connect_timeout, read_timeout):
        opened.append(SimpleNamespace(backend=backend, path=path, body=body, headers=headers))
        return _FakeUpstream(200, body=b'{"answer":"from-peer"}')

    monkeypatch.setattr(S, "open_upstream", fake_open)
    ready = {"primary": True, "embed": True, "rerank": True, "multimodal": True}
    cache = SimpleNamespace(current=lambda: dict(ready))
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg, None, cache, specs))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    host, port = httpd.server_address
    gw = SimpleNamespace(base=f"http://{host}:{port}", opened=opened, ready=ready)
    try:
        yield gw
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_integration_proxied_post_roundtrip(proxy_gateway) -> None:
    req = urllib.request.Request(
        proxy_gateway.base + "/v1/chat/completions",
        data=json.dumps({"model": "senses", "messages": []}).encode(),
        method="POST",
        headers={"Authorization": f"Bearer {_CALLER_TOKEN}"},
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        assert resp.status == 200
        assert resp.headers.get(S.PROXIED_BY_HEADER) == _THOR_ORIGIN
        assert json.loads(resp.read()) == {"answer": "from-peer"}
    call = proxy_gateway.opened[0]
    assert call.backend.base_url == _THOR_ORIGIN
    forwarded = dict(call.headers)
    assert forwarded.get("Authorization") == f"Bearer {_PEER_KEY}"
    assert _CALLER_TOKEN not in json.dumps(list(call.headers))


def test_integration_v1_models_follows_live_peer_signal(proxy_gateway) -> None:
    with urllib.request.urlopen(proxy_gateway.base + "/v1/models", timeout=5) as resp:
        ids = {m["id"] for m in json.loads(resp.read())["data"]}
    assert _SENSES_ID in ids  # peer-ready True → advertised
    proxy_gateway.ready["multimodal"] = False  # peer probe now says down
    with urllib.request.urlopen(proxy_gateway.base + "/v1/models", timeout=5) as resp:
        ids = {m["id"] for m in json.loads(resp.read())["data"]}
    assert _SENSES_ID not in ids  # dropped, exactly like a dead local backend


def test_integration_loop_marked_request_refused(proxy_gateway) -> None:
    req = urllib.request.Request(
        proxy_gateway.base + "/v1/chat/completions",
        data=b'{"model":"senses"}',
        method="POST",
        headers={S.PROXIED_HEADER: "primary"},
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 508
    assert json.loads(exc.value.read())["error"]["type"] == "proxy_loop"
    assert proxy_gateway.opened == []  # zero outbound
