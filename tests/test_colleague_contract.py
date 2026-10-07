"""End-to-end Colleague-contract test (issue #81, task t11).

Proves the honesty conditions ``h1`` + ``h15`` from
``docs/specs/2026-07-03-lobes-exposes-the-full-colleague-runtime-stack-as.md``:

    A Colleague client, given only lobes' machine-readable contract, resolves
    and consumes cortex+senses (plus stt/tts/embedder/reranker) endpoints BY
    ROLE with zero hardcoded model ids; and lobes emits only runtime metrics,
    never task-quality claims.

Two things make this an END-TO-END test rather than a unit test:

1. **The fake fleet** — one loopback ``ThreadingHTTPServer`` that answers
   ``GET /capabilities`` with the REAL production builder
   (:func:`lobes.gateway.server.capabilities_payload`, wired against a real
   :func:`~lobes.gateway._config.build_config`), so the contract under test
   is the actual shipped one, not a hand-rolled fixture — plus canned
   OpenAI-shaped role endpoints (``/v1/chat/completions``,
   ``/v1/embeddings``, ``/v1/rerank``) so the client can actually drive them.
   It never proxies anywhere: unlike the real gateway it answers ``/v1/*``
   itself, which is why the wired ``PRIMARY_URL``/``MULTIMODAL_BASE_URL``/etc
   below point at hosts that don't exist — they are never dialed.

2. **The Colleague client** (:func:`colleague_call_role` +
   :func:`_colleague_request_body`, stdlib-only, ~30 lines total) — given
   ONLY a base URL, it GETs ``/capabilities`` and then issues each request
   using the ``endpoint``/``model``/``path`` it read FROM THE CONTRACT.
   Nothing in those two functions ever spells out a model id literal; this is
   enforced two ways below: statically
   (``test_client_source_never_references_a_hardcoded_model_id`` greps the
   function source for the fake fleet's configured ids) and dynamically
   (``test_colleague_follows_an_operator_rename_with_no_client_code_change``
   reconfigures the operator's served model and shows the SAME client code
   follows the rename with no edits).
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
import socket
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from lobes.embed_lanes import EMBED_LANES
from lobes.gateway import server as S
from lobes.gateway._config import ServerConfig, build_config
from lobes.gateway._routing import Backend
from lobes.roles import ROLES, STT_REALTIME_RESPONSIBILITY, RoleInfo, build_role_registry
from lobes.roles_measure import ALLOWED_METRIC_KEYS, measure_registry

# --- the FAKE FLEET's operator-side configuration ---------------------------
#
# These model ids belong to the fleet's wiring (mirroring a real deployment's
# .env), never to the client. `colleague_call_role` / `_colleague_request_body`
# below must NEVER reference any of these literals — see
# test_client_source_never_references_a_hardcoded_model_id.
_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_MULTIMODAL_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"

# A real vLLM /metrics exposition snippet (mirrors tests/test_cli_measure.py)
# so the LLM-role measure probe gets a genuine mem_usage_pct, not just nulls.
_VLLM_METRICS_TEXT = (
    'vllm:num_requests_running{model_name="x"} 1\n'
    'vllm:num_requests_waiting{model_name="x"} 0\n'
    'vllm:gpu_cache_usage_perc{model_name="x"} 0.42\n'
)


def _free_port() -> int:
    """Reserve an ephemeral loopback port number for the fake fleet.

    ``capabilities_payload`` derives the contract's ``endpoint`` field from
    ``ServerConfig.port`` *before* the server is bound, so the port has to be
    known up front. Mirrors the ``_closed_port()`` helper already used in
    ``tests/test_cli_measure.py`` / ``test_cli_benchmark_profiles.py`` /
    ``test_bench_compare.py`` — bind on port 0, read it back, close, then bind
    the real server on that same number.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _full_env(port: int, **over: str) -> dict[str, str]:
    """A fully-wired four-gateway-role fleet (stt/tts deliberately unset — the
    #81 contract must still enumerate them, present with loaded=False)."""
    env = {
        "GATEWAY_HOST": "127.0.0.1",
        "GATEWAY_PORT": str(port),
        # Never dialed — the fake fleet answers /v1/* itself (see module docstring).
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


# --- the fake fleet: ONE loopback server ------------------------------------
#
# GET /capabilities is answered by the REAL production builder; every other
# route is a minimal canned OpenAI-shaped response so the client can drive it.


class _FakeFleetHandler(BaseHTTPRequestHandler):
    """Bound to a ``(table, cfg, env)`` triple by :func:`_make_fake_fleet_handler`."""

    table = None  # set per-instance-class below (frozen dataclasses → safe to share)
    cfg: ServerConfig
    env: dict

    protocol_version = "HTTP/1.1"

    # --- GET: /capabilities (real), /health + /metrics (canned, for `measure_registry`) ---
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        route = self.path.split("?", 1)[0]
        if route == "/capabilities":
            # THE REAL production builder — the contract under test is the
            # actual shipped one, not a hand-rolled fixture. capabilities_payload
            # never fabricates an endpoint from GATEWAY_HOST/GATEWAY_PORT
            # (issue #81 t5, criterion 3), so — exactly like the real gateway's
            # do_GET route derives `origin` via `reachable_origin` and passes it
            # explicitly — this fake fleet passes its own real, dialable
            # loopback origin (the address this httpd is actually bound to).
            gateway_url = f"http://{self.cfg.host}:{self.cfg.port}"
            self._send_json(
                200,
                S.capabilities_payload(self.table, self.cfg, env=self.env, gateway_url=gateway_url),
            )
        elif route == "/health":
            self._send_json(200, {"status": "ok"})
        elif route == "/metrics":
            self._send_text(200, _VLLM_METRICS_TEXT)
        else:
            self._send_json(404, {"error": {"message": f"not found: {route}"}})

    # --- POST: canned OpenAI-shaped role endpoints ---
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            body = {}
        model = body.get("model")
        route = self.path.split("?", 1)[0]
        if route == "/v1/chat/completions":
            self._send_json(
                200,
                {
                    "id": "chatcmpl-fake",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "canned reply"},
                        }
                    ],
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
                },
            )
        elif route == "/v1/embeddings":
            items = body.get("input") or []
            self._send_json(
                200,
                {
                    "object": "list",
                    "model": model,
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [0.0, 0.1, 0.2]}
                        for i in range(len(items))
                    ],
                    "usage": {"prompt_tokens": 4, "total_tokens": 4},
                },
            )
        elif route == "/v1/rerank":
            docs = body.get("documents") or []
            self._send_json(
                200,
                {
                    "model": model,
                    "results": [
                        {"index": i, "relevance_score": round(1.0 / (i + 1), 3)}
                        for i in range(len(docs))
                    ],
                },
            )
        else:
            self._send_json(404, {"error": {"message": f"not found: {route}"}})

    def _send_json(self, status: int, obj: dict) -> None:
        self._send_bytes(status, "application/json", json.dumps(obj).encode())

    def _send_text(self, status: int, text: str) -> None:
        self._send_bytes(status, "text/plain", text.encode())

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data:
            self.wfile.write(data)

    def log_message(self, fmt: str, *args) -> None:  # keep test output quiet
        pass


def _make_fake_fleet_handler(table, cfg: ServerConfig, env: dict) -> type[_FakeFleetHandler]:
    return type(
        "_BoundFakeFleetHandler", (_FakeFleetHandler,), {"table": table, "cfg": cfg, "env": env}
    )


def _with_lanes(table, lane_names: tuple[str, ...]):
    """``table`` plus a directly-built Backend per named specialist lane.

    The shared t7 contract: a wired lane is a Backend whose ``name`` is the
    lane name, ``served_name`` its catalog id and ``task`` its task family.
    Built here directly so this test pins the contract, not t6's env parsing.
    """
    by_name = {lane.name: lane for lane in EMBED_LANES}
    extra = tuple(
        Backend(
            name=name,
            base_url=f"http://{name}:8000",  # never dialed
            served_name=by_name[name].catalog_id,
            task=by_name[name].task,
        )
        for name in lane_names
    )
    return dataclasses.replace(table, backends=table.backends + extra)


_ALL_LANES = tuple(lane.name for lane in EMBED_LANES)


@contextlib.contextmanager
def _running_fake_fleet(env: dict, lanes: tuple[str, ...] = ()):
    """Stand up the fake fleet for the duration of the ``with`` block."""
    table, cfg = build_config(env)
    if lanes:
        table = _with_lanes(table, lanes)
    handler_cls = _make_fake_fleet_handler(table, cfg, env)
    httpd = ThreadingHTTPServer((cfg.host, cfg.port), handler_cls)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{cfg.host}:{cfg.port}", table, cfg
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def fake_fleet():
    """The default fully-wired fake fleet: ``(base_url, table, cfg, env)``."""
    port = _free_port()
    env = _full_env(port)
    with _running_fake_fleet(env) as (base_url, table, cfg):
        yield base_url, table, cfg, env


@pytest.fixture
def fake_fleet_with_lanes():
    """The fully-wired fake fleet with EVERY specialist lane wired as well."""
    port = _free_port()
    env = _full_env(port)
    with _running_fake_fleet(env, lanes=_ALL_LANES) as (base_url, table, cfg):
        yield base_url, table, cfg, env


# --- the Colleague client ----------------------------------------------------
#
# Stdlib only, ~30 lines total, ZERO hardcoded model ids. Given ONLY a base
# URL, it resolves each role purely from the live /capabilities contract.


def _colleague_request_body(role: str, model: str) -> dict:
    """The OpenAI-shaped request body for ``role``, branched by ROLE FAMILY
    only. ``model`` is threaded straight through from the caller's contract —
    this function never spells out a model id of its own."""
    if role in ("cortex", "senses"):
        return {"model": model, "messages": [{"role": "user", "content": "ping"}]}
    if role == "embedder":
        return {"model": model, "input": ["hello world"]}
    if role == "reranker":
        return {"model": model, "query": "q", "documents": ["doc one", "doc two"]}
    raise ValueError(f"this probe client does not drive role {role!r}")


def colleague_call_role(base_url: str, role: str) -> tuple[dict, dict]:
    """A minimal Colleague client.

    Given ONLY ``base_url``, resolves ``role`` via ``GET /capabilities`` and
    drives it using ONLY the ``endpoint``, ``model``, and ``path`` fields READ
    FROM THE CONTRACT — no model id, endpoint, or path is ever spelled out
    here. Returns ``(contract_entry, response_json)``.
    """
    with urllib.request.urlopen(base_url.rstrip("/") + "/capabilities", timeout=5) as r:
        contract = json.load(r)
    info = contract[role]
    endpoint, model, path = info["endpoint"], info["model"], info["path"]
    body = _colleague_request_body(role, model)
    req = urllib.request.Request(
        endpoint.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return info, json.load(r)


# --- 1. zero hardcoded model ids — enforced at the SOURCE level -------------


def test_client_source_never_references_a_hardcoded_model_id() -> None:
    source = inspect.getsource(colleague_call_role) + inspect.getsource(_colleague_request_body)
    for literal in (_PRIMARY_ID, _MULTIMODAL_ID, _EMBED_ID, _RERANK_ID):
        assert literal not in source, f"client source references a hardcoded model id: {literal!r}"


# --- 2. drives every role BY ROLE via resolved endpoints, zero hardcoded ids,
# proven DYNAMICALLY (not just by source scan) -------------------------------


def test_colleague_drives_cortex_and_senses_via_resolved_endpoints(fake_fleet) -> None:
    base_url, _table, _cfg, _env = fake_fleet
    for role in ("cortex", "senses"):
        info, response = colleague_call_role(base_url, role)
        assert info["loaded"] is True
        assert info["model"]  # resolved to a concrete served name from the contract
        assert response["model"] == info["model"]  # the model actually SENT matches the contract
        assert response["choices"][0]["message"]["content"]


def test_colleague_drives_embedder_and_reranker_via_resolved_endpoints(fake_fleet) -> None:
    base_url, _table, _cfg, _env = fake_fleet
    info, response = colleague_call_role(base_url, "embedder")
    assert info["loaded"] is True
    assert response["model"] == info["model"]
    assert len(response["data"]) == 1  # one input string sent by _colleague_request_body

    info, response = colleague_call_role(base_url, "reranker")
    assert info["loaded"] is True
    assert response["model"] == info["model"]
    assert len(response["results"]) == 2  # two documents sent by _colleague_request_body


def test_colleague_follows_an_operator_rename_with_no_client_code_change() -> None:
    """The strongest proof the client has zero hardcoded ids: rename the
    operator's served cortex model and show the SAME client code resolves and
    drives the new id, with no client edits."""
    port = _free_port()
    env = _full_env(port, PRIMARY_SERVED_NAME="acme/renamed-cortex-9000")
    with _running_fake_fleet(env) as (base_url, _table, _cfg):
        info, response = colleague_call_role(base_url, "cortex")
    assert info["model"] == "acme/renamed-cortex-9000"
    assert response["model"] == "acme/renamed-cortex-9000"


# --- 2b. stt realtime/VAD session capability (issue #149, task t4) ---------
#
# Acceptance criterion 2: a text-only fleet shows no realtime claim. This is
# the negative control against the fake fleet's DEFAULT env — `_full_env`
# above deliberately never sets AUDIO_URL, so stt/tts stay unloaded (see the
# module docstring) — proven here over the REAL production `GET
# /capabilities` builder (`capabilities_payload`), not a hand-rolled fixture.
# The positive side of the same contract (an audio-enabled fleet DOES claim
# it) is asserted end to end in tests/test_cli_capabilities.py; without this
# negative control that positive assertion would be vacuous.


def test_stt_shows_no_realtime_claim_on_text_only_fake_fleet(fake_fleet) -> None:
    base_url, _table, _cfg, _env = fake_fleet
    with urllib.request.urlopen(base_url + "/capabilities", timeout=5) as r:
        contract = json.load(r)
    assert contract["stt"]["loaded"] is False
    assert contract["stt"]["responsibilities"] == ["transcribe", "audio_input_to_text"]
    assert STT_REALTIME_RESPONSIBILITY not in contract["stt"]["responsibilities"]
    assert STT_REALTIME_RESPONSIBILITY not in contract["tts"]["responsibilities"]


def test_stt_advertises_realtime_claim_when_fake_fleet_wires_audio_overlay() -> None:
    """The end-to-end positive companion to the negative control above,
    against the same real production builder: an audio-enabled fake fleet
    (AUDIO_URL wired) DOES claim the capability under stt, over a genuine
    HTTP round trip to GET /capabilities."""
    port = _free_port()
    env = _full_env(port, AUDIO_URL="http://realtime:8080")
    with _running_fake_fleet(env) as (base_url, _table, _cfg):
        with urllib.request.urlopen(base_url + "/capabilities", timeout=5) as r:
            contract = json.load(r)
    assert contract["stt"]["loaded"] is True
    assert STT_REALTIME_RESPONSIBILITY in contract["stt"]["responsibilities"]
    assert STT_REALTIME_RESPONSIBILITY not in contract["tts"]["responsibilities"]


# --- 3. runtime-only boundary (h1): the contract + measure_registry never ---
# emit a task-quality/correctness field --------------------------------------

_QUALITY_TOKENS = (
    "accuracy",
    "correct",
    "quality",
    "task_success",
    "success_rate",
    "grade",
    "score",
)


# Every key a specialist-lane entry carries (orin-embedding-specialist t7).
_LANE_FIELDS = frozenset(
    {
        "lane",
        "model",
        "runtime",
        "endpoint",
        "path",
        "task",
        "context",
        "dimension",
        "mrl_dims",
        "modalities",
        "normalization",
        "tested_on",
        "quant",
        "responsibilities",
        "forbidden_responsibilities",
        "feasible",
        "loaded",
        "ready",
        "fingerprint",
    }
)


def test_capabilities_contract_is_runtime_descriptor_only(fake_fleet_with_lanes) -> None:
    base_url, _table, _cfg, _env = fake_fleet_with_lanes
    with urllib.request.urlopen(base_url + "/capabilities", timeout=5) as r:
        contract = json.load(r)
    known_fields = {f.name for f in dataclasses.fields(RoleInfo)}
    # Every ROLES key is present; the only other keys are wired lanes.
    assert set(ROLES) <= set(contract)
    assert set(contract) - set(ROLES) == set(_ALL_LANES)
    for name, entry in contract.items():
        allowed = _LANE_FIELDS if name in _ALL_LANES else known_fields
        if name in _ALL_LANES:
            assert entry["lane"] is True
            assert _LANE_FIELDS <= set(entry), f"lane {name} is missing a field"
        extra = set(entry) - allowed
        assert not extra, f"{name} carries an undeclared field: {extra}"
        for key in entry:
            lowered = key.lower()
            assert not any(
                tok in lowered for tok in _QUALITY_TOKENS
            ), f"{name}.{key} looks like a task-quality/correctness claim"


def test_measure_registry_emits_only_allowed_runtime_metric_keys(fake_fleet_with_lanes) -> None:
    base_url, table, cfg, env = fake_fleet_with_lanes
    # gateway_url=base_url: the fake fleet's own real, dialable loopback origin
    # — the honest fix for issue #81 t5 criterion 3 (build_role_registry never
    # fabricates an endpoint from GATEWAY_HOST/GATEWAY_PORT), mirroring what the
    # production HTTP route passes via reachable_origin(...).
    registry = build_role_registry(table, cfg, env=env, gateway_url=base_url)
    measured = measure_registry(registry, timeout=3.0)
    # Every ROLES key is present, and a wired lane never leaks into the role
    # registry: lanes are /capabilities keys, never Colleague roles (c35).
    assert set(ROLES) <= set(measured)
    assert not set(measured) & set(_ALL_LANES)
    for role, result in measured.items():
        extra = set(result["metrics"]) - ALLOWED_METRIC_KEYS
        assert not extra, f"{role} emitted a metric key outside ALLOWED_METRIC_KEYS: {extra}"
        for key in result["metrics"]:
            lowered = key.lower()
            assert not any(
                tok in lowered for tok in _QUALITY_TOKENS
            ), f"{role}.metrics.{key} looks like a task-quality/correctness claim"
    # The four gateway-fronted roles — reachable through the fake fleet's
    # canned /health + /v1/* routes — actually came back ready, proving this
    # isn't a vacuous all-null pass.
    for role in ("cortex", "senses", "embedder", "reranker"):
        assert measured[role]["ready"] is True
    for role in ("stt", "tts"):
        assert measured[role]["ready"] is False  # unwired — present, never omitted, never a crash
    # The registry's derived endpoint is the SAME fake fleet the fixture is running.
    assert registry["cortex"].endpoint == base_url


# --- o7: colleague still resolves every seat with lane keys present ---------
#
# Vendored, faithfully, from colleague's own parser — the consumer this
# contract must not break. Source: /home/spark/git/colleague/colleague/lobes.py
# at colleague commit 6081b8c8: ``_parse_role`` (lines 215-258) and the seat
# loop of ``resolve_roles`` (lines 299-326; the urllib fetch above it is
# replaced by the HTTP GET in the test). colleague is not a dependency of this
# repo, so it cannot be imported here; keep this copy in step with upstream.

_COLLEAGUE_RESOLVED_ROLES = ("cortex", "senses")
_COLLEAGUE_OPTIONAL_ROLES = ("stt", "tts", "embedder", "muse", "worker", "associate")


def _colleague_parse_role(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    try:
        model = raw["model"]
        endpoint = raw["endpoint"]
        path = raw["path"]
        context = raw["context"]
        ready = raw["ready"]
        responsibilities = raw["responsibilities"]
        forbidden = raw["forbidden_responsibilities"]

        if not isinstance(model, str) or not model:
            return None
        if not isinstance(endpoint, str) or not isinstance(path, str):
            return None
        if not isinstance(context, int) or isinstance(context, bool):
            return None
        if not isinstance(ready, bool):
            return None
        if not isinstance(responsibilities, list) or not all(
            isinstance(item, str) for item in responsibilities
        ):
            return None
        if not isinstance(forbidden, list) or not all(isinstance(item, str) for item in forbidden):
            return None
        return {"model": model, "endpoint": endpoint, "path": path, "context": context}
    except (KeyError, TypeError):
        return None


def _colleague_resolve_roles(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None
    resolved: dict = {}
    for name in _COLLEAGUE_RESOLVED_ROLES:
        role = _colleague_parse_role(payload.get(name))
        if role is None:
            return None
        resolved[name] = role
    for name in _COLLEAGUE_OPTIONAL_ROLES:
        resolved[name] = _colleague_parse_role(payload.get(name))
    return resolved


def test_colleague_superset_parser_resolves_every_seat_with_lane_keys(
    fake_fleet, fake_fleet_with_lanes
) -> None:
    """o7: the same seats resolve, to the same values, with or without lanes."""

    def _fetch(base_url: str) -> dict:
        with urllib.request.urlopen(base_url + "/capabilities", timeout=5) as r:
            return json.load(r)

    plain = _fetch(fake_fleet[0])
    with_lanes = _fetch(fake_fleet_with_lanes[0])
    assert set(with_lanes) - set(plain) == set(_ALL_LANES)  # lane keys really are there

    resolved_plain = _colleague_resolve_roles(plain)
    resolved = _colleague_resolve_roles(with_lanes)
    assert resolved is not None
    for seat in _COLLEAGUE_RESOLVED_ROLES:
        assert resolved[seat] is not None
    # Every seat — mandatory and optional — resolves exactly as it did before
    # the lane keys existed. The endpoint differs only by the two fleets'
    # ports, so compare everything but it.
    for seat in (*_COLLEAGUE_RESOLVED_ROLES, *_COLLEAGUE_OPTIONAL_ROLES):
        before, after = resolved_plain[seat], resolved[seat]
        assert (before is None) == (after is None), seat
        if after is not None:
            assert {k: v for k, v in after.items() if k != "endpoint"} == {
                k: v for k, v in before.items() if k != "endpoint"
            }, seat
    # The embedder seat still resolves to the ROLE, never to a lane.
    assert resolved["embedder"]["model"] == _EMBED_ID
