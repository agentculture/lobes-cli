"""``_optional_backend`` wiring contract: a backend is real only when its
``*_BASE_URL`` is set (t1, "advertised implies reachable").

Before this change, ``_optional_backend`` wired a backend when EITHER its
``*_BASE_URL`` OR its ``*_SERVED_NAME`` env var was set, falling back to a
hardcoded ``default_url`` naming a compose service that need not actually
exist. On the reference rig this invented two phantom backends —
``multimodal-coder`` and ``middle`` — whose containers were never started:
``GET /v1/models`` advertised them, but every request to them would fail to
connect. The fix: wire a backend only when its ``*_BASE_URL`` is non-empty,
matching the contract the fleet already documents for ``MINOR_BASE_URL``
("empty ⇒ minor silently unwired").
"""

from __future__ import annotations

from lobes.gateway._config import (
    _DEFAULT_MIDDLE,
    _DEFAULT_MULTIMODAL_CODER,
    _DEFAULT_PRIMARY,
    build_config,
)
from lobes.gateway._routing import resolve_model

_MIDDLE_ID = "nvidia/Qwen3-14B-NVFP4"


# --- criterion 1: SERVED_NAME alone (no BASE_URL) wires NEITHER backend -----


def test_served_name_alone_wires_neither_middle_nor_multimodal_coder() -> None:
    table, _ = build_config(
        {
            "MULTIMODAL_CODER_SERVED_NAME": _DEFAULT_MULTIMODAL_CODER,
            "MIDDLE_SERVED_NAME": _MIDDLE_ID,
        }
    )
    names = [b.name for b in table.backends]
    assert "middle" not in names
    assert "multimodal-coder" not in names
    assert names == ["primary"]  # no phantom backends — primary alone


# --- criterion 2: a caller naming the 14B id falls back to default_model,  --
# --- never routed to a "middle" Backend object ------------------------------


def test_resolve_model_for_unwired_middle_id_falls_back_to_default_model() -> None:
    table, _ = build_config({"MIDDLE_SERVED_NAME": _MIDDLE_ID})
    # No backend is actually serving the 14B weights.
    assert not any(b.name == "middle" for b in table.backends)
    assert not any(b.served_name == _MIDDLE_ID for b in table.backends)
    # The caller's request resolves to default_model (the primary's served
    # name) — NOT to the 14B id, and NOT to a middle Backend object (there is
    # none to route to).
    assert table.default_model == _DEFAULT_PRIMARY
    assert resolve_model(table, _MIDDLE_ID) == table.default_model
    assert resolve_model(table, _MIDDLE_ID) == _DEFAULT_PRIMARY


# --- criterion 3: a *_BASE_URL that IS set still wires the backend ----------


def test_middle_base_url_alone_still_wires_the_backend() -> None:
    table, _ = build_config({"MIDDLE_BASE_URL": "http://vllm-middle:8000"})
    mid = next(b for b in table.backends if b.name == "middle")
    assert mid.served_name == _DEFAULT_MIDDLE
    assert mid.base_url == "http://vllm-middle:8000"
    assert resolve_model(table, _DEFAULT_MIDDLE) == _DEFAULT_MIDDLE


def test_multimodal_coder_base_url_alone_still_wires_the_backend() -> None:
    table, _ = build_config({"MULTIMODAL_CODER_BASE_URL": "http://vllm-multimodal-coder:8000"})
    coder = next(b for b in table.backends if b.name == "multimodal-coder")
    assert coder.served_name == _DEFAULT_MULTIMODAL_CODER
    assert coder.base_url == "http://vllm-multimodal-coder:8000"


# --- criterion 4: EMBED_URL / RERANK_URL / PRIMARY_URL are unchanged -------
# (these are the vars the fleet template actually sets, so this is the
# steady-state / most-common-path regression guard for this change)


def test_primary_url_alone_still_wires_primary() -> None:
    table, _ = build_config({"PRIMARY_URL": "http://vllm-primary:8000"})
    assert table.backends[0].name == "primary"
    assert table.backends[0].base_url == "http://vllm-primary:8000"


def test_embed_url_alone_still_wires_embed_backend() -> None:
    table, _ = build_config({"EMBED_URL": "http://vllm-embed:8000"})
    embed = next(b for b in table.backends if b.name == "embed")
    assert embed.base_url == "http://vllm-embed:8000"
    assert embed.task == "embed"


def test_rerank_url_alone_still_wires_rerank_backend() -> None:
    table, _ = build_config({"RERANK_URL": "http://vllm-rerank:8000"})
    rerank = next(b for b in table.backends if b.name == "rerank")
    assert rerank.base_url == "http://vllm-rerank:8000"
    assert rerank.task == "score"


def test_embed_rerank_primary_urls_together_unchanged_from_before() -> None:
    # The standard fleet template shape: three *_URL vars set, no *_SERVED_NAME
    # overrides. Must still produce three backends, exactly as before this change
    # (this scenario never depended on the dropped `or name_key` clause).
    table, _ = build_config(
        {
            "PRIMARY_URL": "http://vllm-primary:8000",
            "EMBED_URL": "http://vllm-embed:8000",
            "RERANK_URL": "http://vllm-rerank:8000",
        }
    )
    names = {b.name for b in table.backends}
    assert names == {"primary", "embed", "rerank"}


# --- criterion 5: enabling a profile that sets *_BASE_URL still wires it ---


def test_middle_profile_with_base_url_and_served_name_both_set_wires_middle() -> None:
    # Mirrors what the compose "middle" profile actually sets: both vars present,
    # BASE_URL doing the wiring, SERVED_NAME customising the served id.
    table, _ = build_config(
        {
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
            "MIDDLE_SERVED_NAME": _MIDDLE_ID,
        }
    )
    mid = next(b for b in table.backends if b.name == "middle")
    assert mid.served_name == _MIDDLE_ID
    assert resolve_model(table, _MIDDLE_ID) == _MIDDLE_ID
