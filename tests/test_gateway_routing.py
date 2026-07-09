"""Pure (no-socket) tests for the gateway: routing, config, and body helpers."""

from __future__ import annotations

from lobes.gateway import server as S
from lobes.gateway._config import _parse_aliases, build_config
from lobes.gateway._routing import (
    Backend,
    RoutingTable,
    is_unknown_model,
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
    # UNSPECIFIED-vs-UNKNOWN is a caller policy (is_unknown_model / handle_post),
    # NOT resolve_model's: resolve_model still maps an unknown non-empty id to
    # default_model so order_backends always has an owner. handle_post 404s the
    # unknown id BEFORE it ever calls resolve_model (honesty h23), so this
    # fall-back is now a pure-routing safety net, not the serving path.
    assert resolve_model(t, "who-knows") == "P"  # unknown → default (routing net)
    assert resolve_model(t, "") == "P"  # empty → default


# --- is_unknown_model: UNKNOWN vs UNSPECIFIED (honesty h23) ----------------


def test_is_unknown_model_distinguishes_unspecified_from_unknown() -> None:
    # UNSPECIFIED (missing/empty) is NOT unknown — it deliberately routes to
    # default_model and must be served, so is_unknown_model is False for it.
    t = _table()
    assert is_unknown_model(t, None) is False  # missing → unspecified, not unknown
    assert is_unknown_model(t, "") is False  # empty → unspecified, not unknown
    # A known alias or any wired backend's served name is KNOWN → not unknown.
    assert is_unknown_model(t, "P") is False  # served name
    assert is_unknown_model(t, "F") is False  # served name
    assert is_unknown_model(t, "fast") is False  # alias → F
    assert is_unknown_model(t, "big") is False  # alias → P
    # A non-empty id that is neither an alias nor any wired served name is UNKNOWN.
    assert is_unknown_model(t, "who-knows") is True
    assert is_unknown_model(t, "never-advertised") is True


def test_is_unknown_model_decided_against_routing_table_not_readiness() -> None:
    # CRITICAL (issue #91): unknown-ness is decided against the ROUTING TABLE
    # (wired backends + aliases), NEVER the readiness-filtered /v1/models list. A
    # backend that is WIRED is KNOWN even if it is dead/warming and thus absent
    # from /v1/models — its served name is still in table.backends. So "F" is
    # known regardless of any readiness verdict; a request for a dead-but-wired F
    # must route (→ 503 owner-down), never 404 as if it were never advertised.
    t = _table()  # F is wired
    assert is_unknown_model(t, "F") is False  # wired → known, dead or not


def test_is_unknown_model_default_model_is_known_even_in_degenerate_table() -> None:
    # The deployment's declared default identity is always KNOWN — naming it
    # explicitly equals leaving `model` unspecified (both route to default). Even
    # a malformed zero-backend table (which nothing serves) must NOT 404 its own
    # default_model: that request stays the terminal 502 malformed-table signal,
    # not a 404. In a well-formed table this is redundant (default_model is a
    # wired served name) — it only guards the pathological case.
    degenerate = RoutingTable(backends=(), default_model="P", aliases={})
    assert is_unknown_model(degenerate, "P") is False  # default → known
    assert is_unknown_model(degenerate, "Q") is True  # anything else → unknown


# --- order_backends -------------------------------------------------------


def test_order_backends_owner_only_no_failover() -> None:
    # No cross-backend failover (issue #91): each served name resolves to
    # exactly its own owner, never to another backend serving a different model.
    t = _table()
    assert [b.name for b in order_backends(t, "P")] == ["primary"]
    assert [b.name for b in order_backends(t, "F")] == ["fallback"]
    # an unmatched served name falls back to the default model's owner — still
    # a single backend, not a chain.
    assert [b.name for b in order_backends(t, "nope")] == ["primary"]


def test_order_backends_always_returns_at_most_one_backend() -> None:
    # The blanket contract (issue #91): order_backends(table, served) has
    # length <= 1 for EVERY input — a known served name, an unknown one, an
    # embed name, a rerank name, or a tier-alias-resolved name. No caller of
    # order_backends should ever have to reason about a failover chain again.
    embed_name = "Qwen/Qwen3-Embedding-0.6B"
    rerank_name = "Qwen/Qwen3-Reranker-0.6B"
    full = RoutingTable(
        backends=(
            Backend("primary", "http://vllm-primary:8000", "P"),
            Backend("fallback", "http://vllm-fallback:8000", "F"),
            Backend("embed", "http://vllm-embed:8000", embed_name, task="embed"),
            Backend("rerank", "http://vllm-rerank:8000", rerank_name, task="score"),
        ),
        default_model="P",
        aliases={"fast": "F", "main": "P"},  # a tier-style alias resolving to "P"
    )
    for requested in ("P", "F", embed_name, rerank_name, "unknown-model", "", None):
        served = resolve_model(full, requested)
        result = order_backends(full, served)
        assert len(result) <= 1, f"{requested!r} -> {served!r} produced {result!r}"
    # A tier alias resolved first through resolve_model must also land on
    # exactly one backend.
    tier_served = resolve_model(full, "main")
    assert len(order_backends(full, tier_served)) <= 1
    # Sanity: known names still resolve to a non-empty (length-1) result — the
    # <= 1 bound above isn't vacuously satisfied by empty lists for real models.
    assert len(order_backends(full, "P")) == 1
    assert len(order_backends(full, embed_name)) == 1
    assert len(order_backends(full, rerank_name)) == 1


def test_order_backends_known_served_returns_its_own_owner() -> None:
    # Acceptance criterion #2: for a known served name, order_backends returns
    # exactly that model's owning backend (not some other backend).
    t = _table()
    assert [b.name for b in order_backends(t, "P")] == ["primary"]
    assert [b.name for b in order_backends(t, "F")] == ["fallback"]


def test_order_backends_unknown_served_returns_default_owner() -> None:
    # Acceptance criterion #3: an unknown served name routes to the
    # default_model's owner (preserving today's "unknown model -> default"
    # behaviour), still a single element.
    t = _table()
    result = order_backends(t, "totally-unknown-model-id")
    assert [b.name for b in result] == ["primary"]
    assert len(result) == 1


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


def test_order_backends_generate_never_failovers_across_models() -> None:
    # INVERTED for issue #91 ("advertised implies reachable"): order_backends
    # used to fail a generate request over to every other same-task backend
    # (e.g. cortex -> multimodal), which meant a dead cortex engine got silently
    # retried against the Gemma backend with a body still naming the Qwen model
    # id -> a terminal, confusing 404 (or worse, a real answer from the wrong
    # model). Now: "P" resolves to primary and ONLY primary, full stop. Fallback
    # and embed/rerank are never attempted for a "P" request.
    t = _full_table()
    result = order_backends(t, "P")
    names = [b.name for b in result]
    assert names == ["primary"]
    assert "fallback" not in names
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
