"""Tier-alias layer + multimodal-backend wiring tests for the gateway (t4, issue #68).

Pure, no-socket tests. The gateway lets a caller request a *capability tier*
instead of a concrete model id; the alias resolves to the served name of the
wired minor / multimodal / primary generate gear respectively, on top of the
existing task-family routing.

Primary vocabulary: ``main``/``minor``/``multimodal``.
Back-compat aliases: ``cheap``→minor / ``normal``→multimodal / ``hard``→primary.

Fallback contract: a tier whose backend is not wired falls back UPWARD to the
nearest available higher-capability tier, ultimately the always-present
``primary`` (so ``multimodal``/``normal``→primary when the multimodal gear is
absent; ``minor``/``cheap``→multimodal else primary when the minor gear is
absent). ``main``/``hard`` are always primary. The same-task constraint is
untouched: an embed/score request never fails over to a generate backend, and
tier aliases apply only to generate.
"""

from __future__ import annotations

from lobes.catalog import TIER_ROLE
from lobes.gateway._config import (
    _DEFAULT_MINOR,
    _DEFAULT_MULTIMODAL,
    _DEFAULT_PRIMARY,
    build_config,
)
from lobes.gateway._routing import (
    Backend,
    order_backends,
    resolve_model,
    tier_aliases,
)

# --- multimodal backend wiring (mirrors the minor backend) -------------------


def test_multimodal_backend_absent_by_default() -> None:
    # No MULTIMODAL_* env → the multimodal generate gear is opt-in and not wired.
    table, _ = build_config({})
    assert not any(b.name == "multimodal" for b in table.backends)


def test_multimodal_base_url_wires_generate_backend_with_default_name() -> None:
    # MULTIMODAL_BASE_URL alone is enough (mirror minor's MINOR_BASE_URL behaviour).
    table, _ = build_config({"MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000/"})
    mm = next(b for b in table.backends if b.name == "multimodal")
    assert mm.task == "generate"
    assert mm.served_name == _DEFAULT_MULTIMODAL
    assert mm.base_url == "http://vllm-multimodal:8000"  # trailing slash stripped


def test_multimodal_served_name_alone_wires_backend() -> None:
    # MULTIMODAL_SERVED_NAME alone wires it with the default URL (mirror minor).
    table, _ = build_config({"MULTIMODAL_SERVED_NAME": "custom/gemma"})
    mm = next(b for b in table.backends if b.name == "multimodal")
    assert mm.base_url == "http://vllm-multimodal:8000"
    assert mm.served_name == "custom/gemma"
    assert mm.task == "generate"


def test_multimodal_default_served_name_is_pinned_gemma_id() -> None:
    # The pinned Gemma 4 12B id is the default when MULTIMODAL_SERVED_NAME is absent.
    table, _ = build_config({"MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000"})
    mm = next(b for b in table.backends if b.name == "multimodal")
    assert mm.served_name == "sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"


# --- primary vocabulary: main / minor / multimodal ---------------------------


def test_main_alias_resolves_to_primary() -> None:
    # model=main always routes to the 27B primary.
    table, _ = build_config({})
    assert resolve_model(table, "main") == _DEFAULT_PRIMARY


def test_minor_alias_resolves_to_minor_gear() -> None:
    # model=minor resolves to the 4B minor gear when it is wired.
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, "minor") == _DEFAULT_MINOR


def test_multimodal_alias_resolves_to_gemma_backend() -> None:
    # model=multimodal resolves to the Gemma 12B gear when it is wired.
    table, _ = build_config({"MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000"})
    assert resolve_model(table, "multimodal") == _DEFAULT_MULTIMODAL


# --- back-compat aliases: cheap / normal / hard ------------------------------


def test_hard_alias_resolves_to_primary() -> None:
    # model=hard (back-compat for main) always routes to the 27B primary.
    table, _ = build_config({})
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_normal_resolves_to_multimodal_back_compat() -> None:
    # normal is the back-compat alias for multimodal; with the Gemma gear wired
    # it must route to the Gemma backend (not the 14B legacy middle).
    table, _ = build_config({"MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000"})
    assert resolve_model(table, "normal") == _DEFAULT_MULTIMODAL


def test_cheap_resolves_to_minor_back_compat() -> None:
    # cheap is the back-compat alias for minor.
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, "cheap") == _DEFAULT_MINOR


# --- tier-alias resolution: full fleet (all three tiers wired) ---------------


def test_three_tier_aliases_resolve_to_their_gears_when_all_wired() -> None:
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        }
    )
    # Back-compat aliases: cheap → 4B minor, normal → 12B multimodal, hard → 27B primary.
    assert resolve_model(table, "cheap") == _DEFAULT_MINOR
    assert resolve_model(table, "normal") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY
    # Primary vocabulary resolves the same gears.
    assert resolve_model(table, "minor") == _DEFAULT_MINOR
    assert resolve_model(table, "multimodal") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "main") == _DEFAULT_PRIMARY


def test_tier_aliases_track_custom_served_names() -> None:
    table, _ = build_config(
        {
            "MINOR_SERVED_NAME": "my/minor",
            "MULTIMODAL_SERVED_NAME": "my/multimodal",
            "PRIMARY_SERVED_NAME": "my/primary",
        }
    )
    assert resolve_model(table, "cheap") == "my/minor"
    assert resolve_model(table, "normal") == "my/multimodal"
    assert resolve_model(table, "hard") == "my/primary"
    assert resolve_model(table, "minor") == "my/minor"
    assert resolve_model(table, "multimodal") == "my/multimodal"
    assert resolve_model(table, "main") == "my/primary"


# --- tier-alias fallback: upward to the nearest available higher tier --------


def test_normal_falls_back_to_primary_when_multimodal_absent() -> None:
    # minor wired, multimodal NOT wired → normal/multimodal escalate UPWARD to primary
    # (no multimodal gear → the next available higher tier is hard/primary).
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, "cheap") == _DEFAULT_MINOR  # minor present
    assert resolve_model(table, "minor") == _DEFAULT_MINOR  # primary vocab
    assert resolve_model(table, "normal") == _DEFAULT_PRIMARY  # multimodal absent → primary
    assert resolve_model(table, "multimodal") == _DEFAULT_PRIMARY  # primary vocab fallback
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_cheap_falls_back_to_multimodal_when_minor_absent() -> None:
    # multimodal wired, minor NOT wired → cheap/minor escalates UPWARD to the
    # multimodal gear (nearest available higher tier), not all the way to primary.
    table, _ = build_config({"MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000"})
    assert resolve_model(table, "cheap") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "minor") == _DEFAULT_MULTIMODAL  # primary vocab fallback
    assert resolve_model(table, "normal") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "multimodal") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY


def test_multimodal_falls_back_to_primary_when_multimodal_absent() -> None:
    # Only minor + primary wired (no multimodal gear) → multimodal/normal → primary
    # (upward, skipping the lower minor tier).
    table, _ = build_config({"MINOR_BASE_URL": "http://vllm-minor:8000"})
    assert resolve_model(table, "multimodal") == _DEFAULT_PRIMARY
    assert resolve_model(table, "normal") == _DEFAULT_PRIMARY


def test_all_tiers_fall_back_to_primary_when_only_primary_wired() -> None:
    # Default fleet (primary alone) → every tier resolves to primary.
    table, _ = build_config({})
    for alias in ("cheap", "normal", "hard", "main", "minor", "multimodal"):
        assert resolve_model(table, alias) == _DEFAULT_PRIMARY


def test_hard_always_resolves_to_primary_even_with_full_fleet() -> None:
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        }
    )
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY
    assert resolve_model(table, "main") == _DEFAULT_PRIMARY


# --- same-task constraint stays intact (tier aliases are generate-only) ------


def test_embed_request_never_fails_over_to_generate_with_tiers_wired() -> None:
    # Full generate fleet (primary + minor + multimodal) PLUS an embed backend.
    # An embed request must resolve to / stay within the embed backend only;
    # the tier aliases must not leak a generate backend in.
    embed_name = "Qwen/Qwen3-Embedding-0.6B"
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
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
    for tier in ("cheap", "normal", "hard", "main", "minor", "multimodal"):
        served = resolve_model(table, tier)
        owner = next(b for b in table.backends if b.served_name == served)
        assert owner.task == "generate"


def test_generate_tier_failover_excludes_pooling_backends() -> None:
    # A generate request still fails over only among generate backends, never to
    # embed/score — even with the multimodal gear added to the generate family.
    embed_name = "Qwen/Qwen3-Embedding-0.6B"
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
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
        Backend("multimodal", "http://mm:8000", "MM"),
        # An embed backend is ignored — tier aliases are generate-only.
        Backend("embed", "http://e:8000", "E", task="embed"),
    )
    aliases = tier_aliases(backends, TIER_ROLE)
    # Primary vocabulary.
    assert aliases["main"] == "P"
    assert aliases["minor"] == "MIN"
    assert aliases["multimodal"] == "MM"
    # Back-compat aliases resolve identically.
    assert aliases["cheap"] == "MIN"
    assert aliases["normal"] == "MM"
    assert aliases["hard"] == "P"


def test_tier_aliases_helper_skips_unwired_and_escalates_upward() -> None:
    # Only the primary generate backend is present → all tiers collapse to it.
    backends = (Backend("primary", "http://p:8000", "P"),)
    aliases = tier_aliases(backends, TIER_ROLE)
    for key in ("main", "minor", "multimodal", "cheap", "normal", "hard"):
        assert aliases[key] == "P", f"expected all tiers → 'P', got {aliases[key]!r} for {key!r}"


def test_tier_aliases_helper_multimodal_absent_escalates_minor_to_multimodal() -> None:
    # minor absent, multimodal present → cheap/minor fall back to multimodal,
    # not all the way to primary.
    backends = (
        Backend("primary", "http://p:8000", "P"),
        Backend("multimodal", "http://mm:8000", "MM"),
    )
    aliases = tier_aliases(backends, TIER_ROLE)
    assert aliases["cheap"] == "MM"  # minor absent → multimodal (next higher)
    assert aliases["minor"] == "MM"
    assert aliases["normal"] == "MM"
    assert aliases["multimodal"] == "MM"
    assert aliases["hard"] == "P"
    assert aliases["main"] == "P"


# --- explicit GATEWAY_ALIASES still coexist with the tier aliases ------------


def test_explicit_gateway_aliases_coexist_with_tier_aliases() -> None:
    table, _ = build_config(
        {
            "MINOR_BASE_URL": "http://vllm-minor:8000",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
            "GATEWAY_ALIASES": "fast=" + _DEFAULT_MINOR,
        }
    )
    # The hand-set alias resolves...
    assert resolve_model(table, "fast") == _DEFAULT_MINOR
    # ...and the tier aliases are still present alongside it.
    assert resolve_model(table, "normal") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "multimodal") == _DEFAULT_MULTIMODAL
    assert resolve_model(table, "hard") == _DEFAULT_PRIMARY
    assert resolve_model(table, "main") == _DEFAULT_PRIMARY
