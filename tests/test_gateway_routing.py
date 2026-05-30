"""Pure (no-socket) tests for the gateway: routing, config, and body helpers."""

from __future__ import annotations

from model_gear.gateway import server as S
from model_gear.gateway._config import _parse_aliases, build_config
from model_gear.gateway._routing import (
    Backend,
    RoutingTable,
    list_models_payload,
    order_backends,
    resolve_model,
    supported_models_payload,
)


def _table() -> RoutingTable:
    return RoutingTable(
        backends=(
            Backend("primary", "http://vllm-primary:8000", "P"),
            Backend("fallback", "http://vllm-fallback:8000", "F"),
        ),
        default_model="P",
        aliases={"fast": "F", "big": "P"},
    )


# --- resolve_model --------------------------------------------------------


def test_resolve_model_exact_alias_default() -> None:
    t = _table()
    assert resolve_model(t, "P") == "P"
    assert resolve_model(t, "F") == "F"
    assert resolve_model(t, "fast") == "F"  # alias
    assert resolve_model(t, "big") == "P"  # alias
    assert resolve_model(t, None) == "P"  # missing → default
    assert resolve_model(t, "who-knows") == "P"  # unknown → default
    assert resolve_model(t, "") == "P"  # empty → default


# --- order_backends -------------------------------------------------------


def test_order_backends_owner_first_then_failover() -> None:
    t = _table()
    assert [b.name for b in order_backends(t, "P")] == ["primary", "fallback"]
    assert [b.name for b in order_backends(t, "F")] == ["fallback", "primary"]
    # an unmatched served name falls back to the default model's owner first
    assert [b.name for b in order_backends(t, "nope")] == ["primary", "fallback"]


def test_list_models_payload_shape() -> None:
    payload = list_models_payload(_table())
    assert payload["object"] == "list"
    assert [m["id"] for m in payload["data"]] == ["P", "F"]
    assert all(m["object"] == "model" for m in payload["data"])


def test_supported_models_payload_annotates_loaded_and_default() -> None:
    t = _table()  # backends serve "P" and "F"; default "P"
    catalog = [
        {"id": "P", "role_hint": "primary", "shape": "dense"},
        {"id": "F", "role_hint": "fallback", "shape": "MoE"},
        {"id": "X", "role_hint": "candidate", "shape": "dense"},  # supported but not loaded
    ]
    payload = supported_models_payload(t, catalog)
    assert payload["object"] == "model-gear.supported_models"
    assert payload["default_model"] == "P"
    by_id = {e["id"]: e for e in payload["data"]}
    assert by_id["P"]["loaded"] is True and by_id["P"]["default"] is True
    assert by_id["F"]["loaded"] is True and by_id["F"]["default"] is False
    assert by_id["X"]["loaded"] is False and by_id["X"]["default"] is False
    assert by_id["F"]["shape"] == "MoE"  # original catalog fields preserved
    assert "loaded" not in catalog[0]  # the input catalog is not mutated


# --- build_config / aliases ----------------------------------------------


def test_build_config_defaults() -> None:
    table, cfg = build_config({})
    assert table.backends[0].served_name == "mmangkad/Qwen3.6-27B-NVFP4"
    assert table.backends[1].served_name == "RedHatAI/Mistral-Small-3.2-24B-Instruct-2506-NVFP4"
    assert table.default_model == "mmangkad/Qwen3.6-27B-NVFP4"  # defaults to primary
    assert table.backends[0].base_url == "http://vllm-primary:8000"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.connect_timeout == 5.0
    assert cfg.read_timeout == 600.0


def test_build_config_overrides_and_url_normalised() -> None:
    table, cfg = build_config(
        {
            "PRIMARY_URL": "http://a:9000/",  # trailing slash stripped
            "PRIMARY_SERVED_NAME": "alpha",
            "FALLBACK_URL": "http://b:9001",
            "FALLBACK_SERVED_NAME": "beta",
            "GATEWAY_DEFAULT_MODEL": "beta",
            "GATEWAY_PORT": "9999",
            "GATEWAY_CONNECT_TIMEOUT": "2.5",
            "GATEWAY_READ_TIMEOUT": "120",
        }
    )
    assert table.backends[0].base_url == "http://a:9000"
    assert table.default_model == "beta"
    assert cfg.port == 9999
    assert cfg.connect_timeout == 2.5
    assert cfg.read_timeout == 120.0


def test_build_config_bad_numbers_fall_back_to_defaults() -> None:
    _, cfg = build_config({"GATEWAY_PORT": "abc", "GATEWAY_READ_TIMEOUT": "nan?"})
    assert cfg.port == 8000
    assert cfg.read_timeout == 600.0


def test_parse_aliases() -> None:
    assert _parse_aliases("a=b, c=d") == {"a": "b", "c": "d"}
    assert _parse_aliases("") == {}
    assert _parse_aliases(None) == {}
    # blank / malformed / half-empty pairs are skipped
    assert _parse_aliases("bad, =x, y=, ok=fine") == {"ok": "fine"}


# --- request-body helpers -------------------------------------------------


def test_extract_model() -> None:
    assert S.extract_model(b'{"model": "foo"}') == "foo"
    assert S.extract_model(b'{"no": "model"}') is None
    assert S.extract_model(b'{"model": 5}') is None  # non-string
    assert S.extract_model(b"not json") is None
    assert S.extract_model(b"[1,2]") is None  # non-dict json


def test_is_streaming() -> None:
    assert S.is_streaming(b'{"stream": true}') is True
    assert S.is_streaming(b'{"stream": false}') is False
    assert S.is_streaming(b'{"x": 1}') is False
    assert S.is_streaming(b"garbage") is False


def test_rewrite_model() -> None:
    import json

    out = S.rewrite_model(b'{"model": "fast", "messages": []}', "served-x")
    assert json.loads(out)["model"] == "served-x"
    assert json.loads(out)["messages"] == []
    # non-JSON / non-dict pass through untouched
    assert S.rewrite_model(b"not json", "x") == b"not json"
    assert S.rewrite_model(b"[1]", "x") == b"[1]"


def test_filter_headers_drops_hop_by_hop() -> None:
    out = dict(
        S.filter_headers(
            [
                ("Host", "x"),
                ("Connection", "keep-alive"),
                ("Content-Length", "10"),
                ("Transfer-Encoding", "chunked"),
                ("Content-Type", "application/json"),
                ("Authorization", "Bearer t"),
            ]
        )
    )
    assert out == {"Content-Type": "application/json", "Authorization": "Bearer t"}


def test_frame_chunk() -> None:
    assert S.frame_chunk(b"hello") == b"5\r\nhello\r\n"
    assert S.frame_chunk(b"") == b"0\r\n\r\n"  # (matches CHUNK_TERMINATOR for empty)


def test_read_chunked_body() -> None:
    import io

    # two chunks + zero terminator; chunk extensions on the first are ignored
    raw = b"5;ext=1\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
    assert S.read_chunked_body(io.BytesIO(raw)) == b"hello world"
    # a malformed size stops the read rather than misreading
    assert S.read_chunked_body(io.BytesIO(b"zz\r\n")) == b""
    # empty / truncated stream → empty body
    assert S.read_chunked_body(io.BytesIO(b"")) == b""
