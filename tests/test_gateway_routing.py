"""Pure (no-socket) tests for the gateway: routing, config, and body helpers."""

from __future__ import annotations

from lobes.gateway import server as S
from lobes.gateway._config import _parse_aliases, build_config
from lobes.gateway._routing import (
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
    assert payload["object"] == "lobes.supported_models"
    assert payload["default_model"] == "P"
    by_id = {e["id"]: e for e in payload["data"]}
    assert by_id["P"]["loaded"] is True and by_id["P"]["default"] is True
    assert by_id["F"]["loaded"] is True and by_id["F"]["default"] is False
    assert by_id["X"]["loaded"] is False and by_id["X"]["default"] is False
    assert by_id["F"]["shape"] == "MoE"  # original catalog fields preserved
    assert "loaded" not in catalog[0]  # the input catalog is not mutated


# --- build_config / aliases ----------------------------------------------


def test_build_config_defaults_single_backend() -> None:
    # Empty env: no embed / rerank / fallback backends opt in, so build_config
    # wires the generate primary alone (the compose default fleet sets EMBED_URL /
    # RERANK_URL, which is exercised separately below).
    table, cfg = build_config({})
    assert len(table.backends) == 1
    assert table.backends[0].served_name == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
    assert table.default_model == "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"  # defaults to primary
    assert table.backends[0].base_url == "http://vllm-primary:8000"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 8000
    assert cfg.connect_timeout == 5.0
    assert cfg.read_timeout == 600.0


def test_build_config_adds_fallback_only_when_configured() -> None:
    # No optional-backend env → just the generate primary.
    assert len(build_config({})[0].backends) == 1
    # FALLBACK_SERVED_NAME alone is NOT enough to wire a second backend — a
    # served name with no URL describes a model, not a reachable backend
    # (see _optional_backend's "advertised implies reachable" contract).
    table, _ = build_config({"FALLBACK_SERVED_NAME": "beta"})
    assert [b.name for b in table.backends] == ["primary"]
    # FALLBACK_URL is what actually wires the backend; FALLBACK_SERVED_NAME
    # alongside it only customises the served name.
    table, _ = build_config(
        {"FALLBACK_URL": "http://vllm-fallback:8000", "FALLBACK_SERVED_NAME": "beta"}
    )
    assert [b.name for b in table.backends] == ["primary", "fallback"]
    assert table.backends[1].served_name == "beta"


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


# --- embed / rerank backends in build_config ---------------------------------


def test_build_config_embed_rerank_backends_added_when_configured() -> None:
    # All four env vars supplied → four backends: primary, embed, rerank.
    # (no fallback vars → no fallback backend)
    embed_name = "Qwen/Qwen3-Embedding-0.6B"
    rerank_name = "Qwen/Qwen3-Reranker-0.6B"
    table, _ = build_config(
        {
            "EMBED_URL": "http://vllm-embed:8000/",
            "EMBED_SERVED_NAME": embed_name,
            "RERANK_URL": "http://vllm-rerank:8000/",
            "RERANK_SERVED_NAME": rerank_name,
        }
    )
    served_names = [b.served_name for b in table.backends]
    assert embed_name in served_names
    assert rerank_name in served_names
    # trailing slash is stripped from URLs
    embed_backend = next(b for b in table.backends if b.name == "embed")
    rerank_backend = next(b for b in table.backends if b.name == "rerank")
    assert embed_backend.base_url == "http://vllm-embed:8000"
    assert rerank_backend.base_url == "http://vllm-rerank:8000"
    # resolve_model routes each served name to itself
    assert resolve_model(table, embed_name) == embed_name
    assert resolve_model(table, rerank_name) == rerank_name


def test_build_config_embed_url_alone_triggers_embed_backend() -> None:
    # EMBED_URL alone (no EMBED_SERVED_NAME) → embed backend added with default name.
    table, _ = build_config({"EMBED_URL": "http://vllm-embed:9999"})
    assert any(b.name == "embed" for b in table.backends)
    embed_backend = next(b for b in table.backends if b.name == "embed")
    assert embed_backend.served_name == "Qwen/Qwen3-Embedding-0.6B"


def test_build_config_rerank_served_name_alone_does_not_wire_rerank_backend() -> None:
    # RERANK_SERVED_NAME alone → no rerank backend (no URL ⇒ nothing reachable).
    table, _ = build_config({"RERANK_SERVED_NAME": "custom/reranker"})
    assert not any(b.name == "rerank" for b in table.backends)
    # RERANK_URL is what wires it; RERANK_SERVED_NAME alongside it customises
    # the served name.
    table, _ = build_config(
        {"RERANK_URL": "http://vllm-rerank:9999", "RERANK_SERVED_NAME": "custom/reranker"}
    )
    rerank_backend = next(b for b in table.backends if b.name == "rerank")
    assert rerank_backend.base_url == "http://vllm-rerank:9999"
    assert rerank_backend.served_name == "custom/reranker"


def test_build_config_no_embed_rerank_vars_leaves_single_primary() -> None:
    # Critical invariant: when none of the embed/rerank/fallback vars are present,
    # build_config produces exactly one backend named "primary" — same as before.
    table, _ = build_config({})
    assert len(table.backends) == 1
    assert table.backends[0].name == "primary"


# --- task-aware failover (t5) -----------------------------------------------

_EMBED_SERVED = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_SERVED = "Qwen/Qwen3-Reranker-0.6B"


def _full_table() -> RoutingTable:
    """A four-backend routing table: primary + fallback (generate), embed, rerank."""
    return RoutingTable(
        backends=(
            Backend("primary", "http://vllm-primary:8000", "P"),
            Backend("fallback", "http://vllm-fallback:8000", "F"),
            Backend("embed", "http://vllm-embed:8000", _EMBED_SERVED, task="embed"),
            Backend("rerank", "http://vllm-rerank:8000", _RERANK_SERVED, task="score"),
        ),
        default_model="P",
        aliases={},
    )


def test_order_backends_embed_returns_only_embed_backend() -> None:
    # An embed request must NOT fail over to generate backends: a chat model
    # returns a confusing 400 for /v1/embeddings.
    t = _full_table()
    result = order_backends(t, _EMBED_SERVED)
    assert len(result) == 1
    assert result[0].name == "embed"
    # Confirm: no generate backend snuck in.
    assert all(b.task == "embed" for b in result)


def test_order_backends_rerank_returns_only_rerank_backend() -> None:
    # A score/rerank request must stay within its own task family.
    t = _full_table()
    result = order_backends(t, _RERANK_SERVED)
    assert len(result) == 1
    assert result[0].name == "rerank"
    assert all(b.task == "score" for b in result)


def test_order_backends_generate_still_failovers_between_generate_backends() -> None:
    # The generate failover contract must not regress: primary owns "P", then
    # falls over to fallback (also generate), but NOT to embed or rerank.
    t = _full_table()
    result = order_backends(t, "P")
    names = [b.name for b in result]
    assert names == ["primary", "fallback"]
    # Embed and rerank must be absent from the generate failover chain.
    assert "embed" not in names
    assert "rerank" not in names


def test_resolve_model_routes_embed_rerank_served_names_to_themselves() -> None:
    # resolve_model must recognise embed/rerank served names as owned and return
    # them unchanged (they are present as served_name in the table).
    t = _full_table()
    assert resolve_model(t, _EMBED_SERVED) == _EMBED_SERVED
    assert resolve_model(t, _RERANK_SERVED) == _RERANK_SERVED


def test_before_state_without_embed_backend_embed_name_hits_generate() -> None:
    # BEFORE-STATE (honesty h4): when no embed backend is configured, a request
    # naming the embed served_name falls through to the default (generate) backend.
    # This is exactly why the embed backend must be configured — without it, an
    # embeddings request silently hits the chat model and gets a 400.
    generate_only = RoutingTable(
        backends=(Backend("primary", "http://vllm-primary:8000", "P"),),
        default_model="P",
        aliases={},
    )
    # resolve_model: unknown name → default (the generate primary's served_name).
    assert resolve_model(generate_only, _EMBED_SERVED) == "P"
    # order_backends: owner is the primary (generate) backend — confirmed fallback
    # to chat model in the embed-backend-absent case.
    result = order_backends(generate_only, _EMBED_SERVED)
    assert result[0].name == "primary"
