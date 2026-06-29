"""Tier-alias layer + middle-backend wiring tests for the gateway (t5, issue #68).

Pure, no-socket tests. The gateway lets a caller request a *capability tier*
(``model=cheap|normal|hard``) instead of a concrete model id; the alias resolves
to the served name of the wired minor / middle / primary generate gear
respectively, on top of the existing task-family routing.

Fallback contract: a tier whose backend is not wired falls back UPWARD to the
nearest available higher tier, ultimately the always-present ``primary`` (so
``normal``→primary when the middle gear is absent; ``hard`` is always primary).
The same-task constraint is untouched: an embed/score request never fails over
to a generate backend, and tier aliases apply only to generate.
"""

from __future__ import annotations

from lobes.catalog import TIER_ROLE
from lobes.gateway._config import (
    _DEFAULT_MIDDLE,
    _DEFAULT_MINOR,
    _DEFAULT_PRIMARY,
    build_config,
)
from lobes.gateway._routing import (
    Backend,
    order_backends,
    resolve_model,
    tier_aliases,
)

# --- middle backend wiring (mirrors the minor backend) -----------------------


def test_middle_backend_absent_by_default() -> None:
    # No MIDDLE_* env → the middle generate gear is opt-in and not wired.
    table, _ = build_config({})
    assert not any(b.name == "middle" for b in table.backends)


def test_middle_base_url_wires_generate_backend_with_default_name() -> None:
    # MIDDLE_BASE_URL alone is enough (mirror minor's MINOR_BASE_URL behaviour).
    table, _ = build_config({"MIDDLE_BASE_URL": "http://vllm-middle:8000/"})
    middle = next(b for b in table.backends if b.name == "middle")
    assert middle.task == "generate"
    assert middle.served_name == _DEFAULT_MIDDLE
    assert middle.base_url == "http://vllm-middle:8000"  # trailing slash stripped


def test_middle_served_name_alone_wires_backend() -> None:
    # MIDDLE_SERVED_NAME alone wires it with the default URL (mirror minor).
    table, _ = build_config({"MIDDLE_SERVED_NAME": "custom/14b"})
    middle = next(b for b in table.backends if b.name == "middle")
    assert middle.base_url == "http://vllm-middle:8000"
    assert middle.served_name == "custom/14b"
    assert middle.task == "generate"


# --- tier-alias resolution: full fleet --------------------------------------


def test_three_tier_aliases_resolve_to_their_gears_when_all_wired() -> None:
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
        }
    )
    # cheap → 4B minor, normal → 14B middle, hard → 27B primary.
    assert resolve_model(table, "cheap") == _DEFAULT_MINOR
    assert resolve_model(table, "normal") == _DEFAULT_MIDDLE
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_tier_aliases_track_custom_served_names() -> None:
    table, _ = build_config(
        {
            "MINOR_SERVED_NAME": "my/minor",
            "MIDDLE_SERVED_NAME": "my/middle",
            "PRIMARY_SERVED_NAME": "my/primary",
        }
    )
    assert resolve_model(table, "cheap") == "my/minor"
    assert resolve_model(table, "normal") == "my/middle"
    assert resolve_model(table, "hard") == "my/primary"


# --- tier-alias fallback: upward to the nearest available higher tier --------


def test_normal_falls_back_to_primary_when_middle_absent() -> None:
    # minor wired, middle NOT wired → normal escalates UPWARD to primary
    # (no middle gear → the next available higher tier is hard/primary).
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, "cheap") == _DEFAULT_MINOR  # minor present
    assert resolve_model(table, "normal") == _DEFAULT_PRIMARY  # middle absent → primary
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_cheap_falls_back_to_middle_when_minor_absent() -> None:
    # middle wired, minor NOT wired → cheap escalates UPWARD to the middle gear
    # (nearest available higher tier), not all the way to primary.
    table, _ = build_config({"MIDDLE_BASE_URL": "http://vllm-middle:8000"})
    assert resolve_model(table, "cheap") == _DEFAULT_MIDDLE
    assert resolve_model(table, "normal") == _DEFAULT_MIDDLE
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_all_tiers_fall_back_to_primary_when_only_primary_wired() -> None:
    # Default fleet (primary alone) → every tier resolves to primary.
    table, _ = build_config({})
    assert resolve_model(table, "cheap") == _DEFAULT_PRIMARY
    assert resolve_model(table, "normal") == _DEFAULT_PRIMARY
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_hard_always_resolves_to_primary_even_with_full_fleet() -> None:
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
        }
    )
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


# --- same-task constraint stays intact (tier aliases are generate-only) ------


def test_embed_request_never_fails_over_to_generate_with_tiers_wired() -> None:
    # Full generate fleet (primary + minor + middle) PLUS an embed backend.
    # An embed request must resolve to / stay within the embed backend only;
    # the cheap/normal/hard tier aliases must not leak a generate backend in.
    embed_name = "Qwen/Qwen3-Embedding-0.6B"
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
            "EMBED_URL": "http://vllm-embed:8000",
            "EMBED_SERVED_NAME": embed_name,
        }
    )
    # The embed served name routes to itself (it owns it), never to a tier alias.
    assert resolve_model(table, embed_name) == embed_name
    result = order_backends(table, embed_name)
    assert [b.name for b in result] == ["embed"]
    assert all(b.task == "embed" for b in result)
    # Sanity: the tier aliases all point at generate served names.
    for tier in ("cheap", "normal", "hard"):
        served = resolve_model(table, tier)
        owner = next(b for b in table.backends if b.served_name == served)
        assert owner.task == "generate"


def test_generate_tier_failover_excludes_pooling_backends() -> None:
    # A generate request still fails over only among generate backends, never to
    # embed/score — even with the middle gear added to the generate family.
    embed_name = "Qwen/Qwen3-Embedding-0.6B"
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
            "EMBED_URL": "http://vllm-embed:8000",
            "EMBED_SERVED_NAME": embed_name,
        }
    )
    names = [b.name for b in order_backends(table, _DEFAULT_PRIMARY)]
    assert names[0] == "primary"
    assert "embed" not in names
    assert all(b.task == "generate" for b in order_backends(table, _DEFAULT_PRIMARY))


# --- pure tier_aliases helper -----------------------------------------------


def test_tier_aliases_helper_is_pure_and_uses_backend_role_names() -> None:
    backends = (
        Backend("primary", "http://p:8000", "P"),
        Backend("minor", "http://m:8000", "MIN"),
        Backend("middle", "http://mid:8000", "MID"),
        # An embed backend is ignored — tier aliases are generate-only.
        Backend("embed", "http://e:8000", "E", task="embed"),
    )
    aliases = tier_aliases(backends, TIER_ROLE)
    assert aliases == {"cheap": "MIN", "normal": "MID", "hard": "P"}


def test_tier_aliases_helper_skips_unwired_and_escalates_upward() -> None:
    # Only the primary generate backend is present → all tiers collapse to it.
    backends = (Backend("primary", "http://p:8000", "P"),)
    assert tier_aliases(backends, TIER_ROLE) == {
        "cheap": "P",
        "normal": "P",
        "hard": "P",
    }


# --- explicit GATEWAY_ALIASES still coexist with the tier aliases ------------


def test_explicit_gateway_aliases_coexist_with_tier_aliases() -> None:
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MIDDLE_BASE_URL": "http://vllm-middle:8000",
            "GATEWAY_ALIASES": "fast=" + _DEFAULT_MINOR,
        }
    )
    # The hand-set alias resolves...
    assert resolve_model(table, "fast") == _DEFAULT_MINOR
    # ...and the tier aliases are still present alongside it.
    assert resolve_model(table, "normal") == _DEFAULT_MIDDLE
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY
