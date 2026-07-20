"""The opt-in ``embed-deep`` gear: a SECOND ``task="embed"`` backend.

The embed lane historically wired exactly one backend, so these tests pin the
two properties that make a second one safe:

1. **Opt-in on ``EMBED_DEEP_BASE_URL``** — unset ⇒ no backend, no alias, so an
   existing deployment renders byte-identically (the ``*_BASE_URL`` contract
   ``test_gateway_config_wiring`` pins for middle/multimodal-coder).
2. **No fallback, ever** — unlike a generate-lane capability tier (which falls
   back *upward* via ``tier_aliases``), an absent deep gear must NOT degrade to
   the 0.6B. The two embedders occupy different vector spaces, so a silent
   downgrade would answer with meaningless similarity rather than fail. This is
   the one behaviour a future refactor is most likely to break by "helpfully"
   generalising the tier-fallback machinery over the embed lane.
"""

from __future__ import annotations

from lobes.gateway._config import _DEFAULT_EMBED, _DEFAULT_EMBED_DEEP, build_config
from lobes.gateway._routing import order_backends, resolve_model

_EMBED_URL = "http://vllm-embed:8000"
_DEEP_URL = "http://vllm-embed-deep:8000"


# --- opt-in wiring ----------------------------------------------------------


def test_unset_base_url_wires_no_deep_backend_and_no_alias() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL})
    assert "embed-deep" not in [b.name for b in table.backends]
    assert "embed-deep" not in table.aliases


def test_served_name_alone_does_not_wire_the_deep_backend() -> None:
    # Mirrors the middle/multimodal-coder phantom-backend contract: a served
    # name with no URL describes a model, not a reachable backend.
    table, _ = build_config(
        {"EMBED_URL": _EMBED_URL, "EMBED_DEEP_SERVED_NAME": _DEFAULT_EMBED_DEEP}
    )
    assert "embed-deep" not in [b.name for b in table.backends]
    assert "embed-deep" not in table.aliases


def test_base_url_wires_the_deep_backend_on_the_embed_task() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    deep = next(b for b in table.backends if b.name == "embed-deep")
    assert deep.base_url == _DEEP_URL
    assert deep.served_name == _DEFAULT_EMBED_DEEP
    assert deep.task == "embed"


def test_both_embed_gears_coexist_as_distinct_backends() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    embed_backends = {b.name: b for b in table.backends if b.task == "embed"}
    assert set(embed_backends) == {"embed", "embed-deep"}
    # Distinct served names is what keeps served-name routing unambiguous.
    assert embed_backends["embed"].served_name == _DEFAULT_EMBED
    assert embed_backends["embed-deep"].served_name == _DEFAULT_EMBED_DEEP


# --- the alias --------------------------------------------------------------


def test_alias_resolves_to_the_deep_served_name_when_wired() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    assert table.aliases["embed-deep"] == _DEFAULT_EMBED_DEEP
    assert resolve_model(table, "embed-deep") == _DEFAULT_EMBED_DEEP


def test_alias_honours_a_custom_served_name() -> None:
    # The slot is named for its job, not its checkpoint — an operator may fill
    # it with a different model and the alias must follow.
    table, _ = build_config(
        {
            "EMBED_URL": _EMBED_URL,
            "EMBED_DEEP_BASE_URL": _DEEP_URL,
            "EMBED_DEEP_SERVED_NAME": "Qwen/Qwen3-Embedding-8B",
        }
    )
    assert table.aliases["embed-deep"] == "Qwen/Qwen3-Embedding-8B"


def test_alias_routes_to_the_deep_backend_not_the_hot_path_gear() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    ordered = order_backends(table, resolve_model(table, "embed-deep"))
    assert [b.name for b in ordered] == ["embed-deep"]


def test_the_hot_path_embedder_is_unaffected_by_the_deep_gear() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    ordered = order_backends(table, resolve_model(table, _DEFAULT_EMBED))
    assert [b.name for b in ordered] == ["embed"]


# --- the no-fallback invariant (the vector-space safety property) -----------


def test_unwired_deep_alias_does_not_fall_back_to_the_shallow_embedder() -> None:
    # THE load-bearing assertion. tier_aliases falls back upward for generate
    # tiers; the embed lane must not, because the 0.6B would answer from a
    # different vector space — meaningless scores instead of an honest failure.
    table, _ = build_config({"EMBED_URL": _EMBED_URL})
    assert resolve_model(table, "embed-deep") != _DEFAULT_EMBED
    # Unknown model ⇒ the table's default (the primary), which callers surface
    # as an error rather than silently treating as an embedding answer.
    assert resolve_model(table, "embed-deep") == table.default_model


def test_deep_alias_is_absent_when_only_the_shallow_gear_is_wired() -> None:
    table, _ = build_config({"EMBED_URL": _EMBED_URL})
    assert table.aliases.get("embed-deep") is None


def test_deep_gear_wires_even_without_the_shallow_gear() -> None:
    # The two are independent opt-ins; neither implies the other.
    table, _ = build_config({"EMBED_DEEP_BASE_URL": _DEEP_URL})
    names = [b.name for b in table.backends if b.task == "embed"]
    assert names == ["embed-deep"]


# --- it is a gear, not a role ----------------------------------------------


def test_deep_gear_has_no_feasibility_or_peer_channel() -> None:
    # embed-deep shares the embedder role's responsibility contract, so it gets
    # no RoleProfile, no *_FEASIBLE, and no peer referral/proxy knobs — exactly
    # like every other opt-in gear (minor / middle / multimodal-coder).
    from lobes.gateway._config import (
        FEASIBLE_ENV,
        PEER_API_KEY_ENV,
        PEER_ORIGIN_ENV,
        PEER_PROXY_ENV,
    )

    for table_ in (FEASIBLE_ENV, PEER_ORIGIN_ENV, PEER_PROXY_ENV, PEER_API_KEY_ENV):
        assert "embed-deep" not in table_

    table, _ = build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    assert "embed-deep" not in table.infeasible
