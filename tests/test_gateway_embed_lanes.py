"""Specialist embed/rerank lanes wired from the registry (orin-embedding-specialist t6).

Every lane in :data:`lobes.embed_lanes.EMBED_LANES` gets ONE optional gateway
backend, wired only behind its own ``<LANE>_BASE_URL`` and addressed by its own
name or its raw checkpoint id. The property these tests pin hardest is the one
a "helpful" refactor is most likely to break: **no fallback across lanes,
ever** (o5). Two embedders occupy different vector spaces, so an unwired,
infeasible or stopped lane must answer 404/503 — never another lane, another
checkpoint, or the 0.6B default embedder.

The multimodal pass-through test (o6) asserts the exact bytes the fake
upstream receives.
"""

from __future__ import annotations

import base64
import json

import pytest

from lobes.embed_lanes import EMBED_LANES
from lobes.gateway import server as S
from lobes.gateway._config import _DEFAULT_EMBED, build_config, lane_env_key
from lobes.gateway._routing import order_backends, resolve_model

_EMBED_URL = "http://vllm-embed:8000"
_LANES = {lane.name: lane for lane in EMBED_LANES}


def _url(lane_name: str) -> str:
    return f"http://{lane_name}:8000"


def _env(*wired: str, **extra: str) -> dict[str, str]:
    env = {"EMBED_URL": _EMBED_URL}
    for name in wired:
        env[_LANES[name].base_url_env] = _url(name)
    env.update(extra)
    return env


class _FakeUpstream:
    def __init__(self, status: int = 200, body: bytes = b'{"ok":1}') -> None:
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body

    def read_all(self) -> bytes:
        return self._body

    def read(self, _n: int) -> bytes:
        data, self._body = self._body, b""
        return data

    def close(self) -> None:
        pass


def _opener(down: frozenset[str] = frozenset()):
    """An ``open_upstream`` stub recording (backend name, body) per dial.

    Every backend answers 200 unless named in ``down``, in which case the dial
    raises :class:`~lobes.gateway.server.UpstreamError` (connection refused).
    """
    calls: list[tuple[str, bytes]] = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, body))
        if backend.name in down:
            raise S.UpstreamError(f"{backend.name}: connection refused")
        return _FakeUpstream()

    return opener, calls


def _post(env: dict[str, str], body: bytes, *, path="/v1/embeddings", down=frozenset()):
    table, cfg = build_config(env)
    opener, calls = _opener(down)
    resp = S.handle_post(table, cfg, path, [], body, opener)
    return resp, calls


def _error_code(resp) -> str:
    error = json.loads(resp.body)["error"]
    return error.get("code") or error["type"]


# --- registry-driven wiring --------------------------------------------------


def test_lane_env_keys_follow_the_shared_contract() -> None:
    assert lane_env_key("gemma2-embed", "BASE_URL") == "GEMMA2_EMBED_BASE_URL"
    assert lane_env_key("nomic-code-embed", "FEASIBLE") == "NOMIC_CODE_EMBED_FEASIBLE"
    assert lane_env_key("qwen3vl-rerank", "MAX_ACTIVE") == "QWEN3VL_RERANK_MAX_ACTIVE"
    for lane in EMBED_LANES:
        assert lane_env_key(lane.name, "BASE_URL") == lane.base_url_env


def test_no_lane_env_wires_no_lane_backend_and_no_alias() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL})
    names = {b.name for b in table.backends}
    for lane in EMBED_LANES:
        assert lane.name not in names
        assert lane.name not in table.aliases


@pytest.mark.parametrize("lane", EMBED_LANES, ids=lambda lane: lane.name)
def test_each_lane_wires_behind_its_own_base_url(lane) -> None:
    table, _ = build_config(_env(lane.name))
    backend = next(b for b in table.backends if b.name == lane.name)
    assert backend.base_url == _url(lane.name)
    assert backend.served_name == lane.catalog_id
    assert backend.task == lane.task
    # Only this lane is wired — never its siblings.
    others = {other.name for other in EMBED_LANES} - {lane.name}
    assert not others & {b.name for b in table.backends}


@pytest.mark.parametrize("lane", EMBED_LANES, ids=lambda lane: lane.name)
def test_lane_name_and_raw_checkpoint_id_both_resolve_to_the_lane(lane) -> None:
    table, _ = build_config(_env(*_LANES))
    assert table.aliases[lane.name] == lane.catalog_id
    for requested in (lane.name, lane.catalog_id):
        served = resolve_model(table, requested)
        assert served == lane.catalog_id
        assert [b.name for b in order_backends(table, served)] == [lane.name]


def test_trailing_slash_on_base_url_is_stripped() -> None:
    table, _ = build_config({"GEMMA2_EMBED_BASE_URL": "http://gemma2-embed:8000/"})
    backend = next(b for b in table.backends if b.name == "gemma2-embed")
    assert backend.base_url == "http://gemma2-embed:8000"


def test_wiring_a_lane_leaves_embed_and_embed_deep_untouched() -> None:
    base = {"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": "http://vllm-embed-deep:8000"}
    before, _ = build_config(base)
    after, _ = build_config({**base, "GEMMA2_EMBED_BASE_URL": _url("gemma2-embed")})
    pick = ("embed", "embed-deep")
    assert [b for b in before.backends if b.name in pick] == [
        b for b in after.backends if b.name in pick
    ]
    assert after.aliases["embed-deep"] == before.aliases["embed-deep"]
    assert after.aliases["embedder"] == _DEFAULT_EMBED


# --- MAX_ACTIVE and FEASIBLE per lane ---------------------------------------


def test_lane_max_active_parses_per_backend() -> None:
    _, cfg = build_config(
        _env("gemma2-embed", "nemotron-embed", GEMMA2_EMBED_MAX_ACTIVE="4", EMBED_MAX_ACTIVE="2")
    )
    assert cfg.local_capacities["gemma2-embed"] == 4.0
    assert cfg.local_capacities["embed"] == 2.0
    assert "nemotron-embed" not in cfg.local_capacities


def test_malformed_lane_max_active_raises_loudly() -> None:
    from lobes.gateway._config import CapacityConfigError

    env = _env("gemma2-embed", GEMMA2_EMBED_MAX_ACTIVE="lots")
    with pytest.raises(CapacityConfigError, match="GEMMA2_EMBED_MAX_ACTIVE"):
        build_config(env)


def test_lane_feasible_false_marks_only_that_lane_infeasible() -> None:
    table, _ = build_config(_env("gemma2-embed", "nemotron-embed", GEMMA2_EMBED_FEASIBLE="false"))
    assert "gemma2-embed" in table.infeasible
    assert "nemotron-embed" not in table.infeasible


# --- o5: never served by another lane ---------------------------------------


def _embed_body(model: str) -> bytes:
    return json.dumps({"model": model, "input": ["def add(a, b): return a + b"]}).encode()


@pytest.mark.parametrize("lane", EMBED_LANES, ids=lambda lane: lane.name)
def test_unwired_lane_404s_and_dials_nothing(lane) -> None:
    # Every OTHER lane, the 0.6B embedder and embed-deep are wired and healthy:
    # a leak to any of them would answer 200.
    others = [name for name in _LANES if name != lane.name]
    env = _env(*others, EMBED_DEEP_BASE_URL="http://vllm-embed-deep:8000")
    for requested in (lane.name, lane.catalog_id):
        resp, calls = _post(env, _embed_body(requested))
        assert resp.status == 404
        assert _error_code(resp) == "model_not_found"
        assert calls == []


def test_infeasible_lane_404s_role_infeasible_and_dials_nothing() -> None:
    env = _env(*_LANES, GEMMA2_EMBED_FEASIBLE="false")
    for requested in ("gemma2-embed", "google/embeddinggemma-2"):
        resp, calls = _post(env, _embed_body(requested))
        assert resp.status == 404
        assert _error_code(resp) == "role_infeasible"
        assert calls == []


@pytest.mark.parametrize("lane", EMBED_LANES, ids=lambda lane: lane.name)
def test_stopped_lane_503s_after_dialling_only_itself(lane) -> None:
    path = "/v1/score" if lane.task == "score" else "/v1/embeddings"
    resp, calls = _post(_env(*_LANES), _embed_body(lane.name), path=path, down={lane.name})
    assert resp.status == 503
    assert _error_code(resp) == "backend_unavailable"
    assert [name for name, _ in calls] == [lane.name]


def test_wired_lane_serves_from_its_own_backend_only() -> None:
    resp, calls = _post(_env(*_LANES), _embed_body("nomic-code-embed"))
    assert resp.status == 200
    assert [name for name, _ in calls] == ["nomic-code-embed"]


# --- o6: a multimodal messages body reaches the upstream byte-identical ------


def _multimodal_body(model: str) -> bytes:
    png = base64.b64encode(bytes(range(256)) * 4).decode()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a red square"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png}"}},
                    {"type": "video_url", "video_url": {"url": "file:///clips/a.mp4"}},
                ],
            }
        ],
        "encoding_format": "float",
    }
    # A compact, non-default serialisation with a non-ASCII character: a gateway
    # that re-serialises the JSON would change these bytes.
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode() + b" \n"


@pytest.mark.parametrize("lane_name", ["qwen3vl-embed", "gemma2-embed"])
def test_multimodal_messages_body_reaches_upstream_byte_identical(lane_name) -> None:
    body = _multimodal_body(_LANES[lane_name].catalog_id).replace(b"red", "réd".encode())
    resp, calls = _post(_env(*_LANES), body)
    assert resp.status == 200
    assert calls == [(lane_name, body)]


def test_multimodal_messages_survive_alias_rewrite_intact() -> None:
    body = _multimodal_body("qwen3vl-embed")
    resp, calls = _post(_env(*_LANES), body)
    assert resp.status == 200
    [(name, sent)] = calls
    assert name == "qwen3vl-embed"
    sent_json, orig_json = json.loads(sent), json.loads(body)
    assert sent_json["model"] == "Qwen/Qwen3-VL-Embedding-8B"
    assert sent_json["messages"] == orig_json["messages"]
    assert {k: v for k, v in sent_json.items() if k != "model"} == {
        k: v for k, v in orig_json.items() if k != "model"
    }
