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


# --- `lobes switch` must not misdirect the deep gear at the hot path --------


def test_switch_notice_names_the_deep_service_for_the_deep_gear() -> None:
    """A switch notice naming ``vllm-embed`` for the 4B would tell an operator to
    replace the 0.6B IN PLACE — silently invalidating every vector in an index
    built with it. Found by an independent review pass; see the docstring on
    ``_pooling_notice``.
    """
    from lobes.catalog import SUPPORTED_MODELS
    from lobes.cli._commands.switch import _pooling_notice

    deep = next(m for m in SUPPORTED_MODELS if m.id == "Qwen/Qwen3-Embedding-4B")
    notice = _pooling_notice(deep)
    assert notice is not None
    assert "vllm-embed-deep" in notice
    # The bare hot-path service name must not appear as a standalone word.
    assert "vllm-embed service" not in notice


def test_switch_notice_still_names_the_hot_path_service_for_the_0_6b() -> None:
    from lobes.catalog import SUPPORTED_MODELS
    from lobes.cli._commands.switch import _pooling_notice

    hot = next(m for m in SUPPORTED_MODELS if m.id == "Qwen/Qwen3-Embedding-0.6B")
    notice = _pooling_notice(hot)
    assert notice is not None
    assert "vllm-embed service" in notice
    assert "vllm-embed-deep" not in notice


def test_switch_notice_unchanged_for_the_reranker() -> None:
    from lobes.catalog import SUPPORTED_MODELS
    from lobes.cli._commands.switch import _pooling_notice

    rr = next(m for m in SUPPORTED_MODELS if m.task == "score")
    notice = _pooling_notice(rr)
    assert notice is not None
    assert "vllm-rerank" in notice


# --- review findings (PR #148) ----------------------------------------------


def test_switch_gives_the_deep_gear_its_own_pooling_budget() -> None:
    """The shared 0.06 pooling default is SMALLER than the 4B's own weights.

    Measured on the GB10: weights 7.56 GiB vs a 0.06 x 121.69 = 7.30 GiB budget, so
    `lobes switch Qwen/Qwen3-Embedding-4B` under the shared default could not load
    the model at all. Raised by Qodo on PR #148.
    """
    from lobes.cli._commands.switch import POOLING_DEFAULT_UTIL, _pooling_default_util

    assert _pooling_default_util("Qwen/Qwen3-Embedding-4B") == 0.11
    # the hot-path gear and anything uncatalogued keep the shared default
    assert _pooling_default_util("Qwen/Qwen3-Embedding-0.6B") == POOLING_DEFAULT_UTIL
    assert _pooling_default_util("some/uncatalogued-embedder") == POOLING_DEFAULT_UTIL


def test_catalogued_pooling_budget_covers_the_model_weights() -> None:
    """Any pooling gear's budget must exceed its own weights, or it cannot boot."""
    from lobes.catalog import SUPPORTED_MODELS
    from lobes.cli._commands.switch import _pooling_default_util

    gb10_total_gib = 121.69  # torch-reported on the reference card
    # (model id, measured weight GiB) for pooling gears we have booted
    measured = {"Qwen/Qwen3-Embedding-4B": 7.56}
    for model in SUPPORTED_MODELS:
        if model.task not in ("embed", "score") or model.id not in measured:
            continue
        budget = _pooling_default_util(model.id) * gb10_total_gib
        assert budget > measured[model.id], (
            f"{model.id}: pooling budget {budget:.2f} GiB does not cover its "
            f"{measured[model.id]} GiB of weights"
        )


def test_colliding_served_names_warn_on_stderr(capsys) -> None:
    """Two backends sharing a served name make routing order-dependent.

    Only an operator can cause it, so the gateway still starts — but silence here
    would defeat embed-deep's whole no-fallback guarantee, because the wrong owner
    answers from a different vector space. Raised by Qodo on PR #148.
    """
    build_config(
        {
            "EMBED_URL": _EMBED_URL,
            "EMBED_DEEP_BASE_URL": _DEEP_URL,
            "EMBED_DEEP_SERVED_NAME": _DEFAULT_EMBED,  # collides with the 0.6B
        }
    )
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert _DEFAULT_EMBED in err
    assert "embed-deep" in err
    assert "VECTOR SPACE" in err


def test_distinct_served_names_warn_nothing(capsys) -> None:
    build_config({"EMBED_URL": _EMBED_URL, "EMBED_DEEP_BASE_URL": _DEEP_URL})
    assert "WARNING" not in capsys.readouterr().err


def test_memory_skill_wrappers_point_at_the_gateway_not_the_dead_port() -> None:
    """The wrappers forced EIDETIC_EMBED_URL to :8002 — a port nothing listens on.

    That silently forced every semantic recall onto eidetic's 128-dim lexical-hash
    fallback while the SKILL.md docs claimed otherwise. Raised by Qodo on PR #148
    as a doc-vs-script mismatch; it was really a live functional break.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in (
        ".claude/skills/recall/scripts/recall.sh",
        ".claude/skills/remember/scripts/remember.sh",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "http://localhost:8002/v1" not in text, f"{rel}: still on the dead port"
        assert "${EIDETIC_EMBED_URL:=http://localhost:8001/v1}" in text, rel
