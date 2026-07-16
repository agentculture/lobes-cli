"""Tests for the THIRD capabilities honesty state — PROXIED (proxy-lobes t5,
issues #115/#127).

Honesty table from #115: ``awake`` = hosted + ready | ``asleep`` = flagged not
hosted + referral | ``proxy`` = advertised as PROXIED (origin = the peer),
NEVER as locally served. t1 (already merged into this branch's base) added
the config channel — :attr:`~lobes.gateway._routing.RoutingTable.peer_proxied`
— but left the capabilities payload untouched (config only, "nothing dials
these peers yet"). This module adds the ``"proxied": true`` marker to both
capabilities surfaces (the gateway's ``GET /capabilities`` and the CLI's
offline fallback both funnel through the ONE shared
:func:`lobes.roles.annotate_peer_referrals`, so testing that one function
here covers both surfaces).

Three states, told apart by KEY PRESENCE alone (never a sentinel value):

* **hosted**        — neither ``hosted_by`` nor ``proxied`` present.
* **referral-only**  — ``hosted_by`` present, ``proxied`` ABSENT.
* **proxied**        — ``hosted_by`` present, ``proxied: true`` ALSO present.

``feasible`` stays ``false`` for a proxied role (still a hardware/deployment
fact), and ``ready`` is never hardcoded ``true`` for one — it stays sourced
from the same clamp/backend_ready machinery :func:`build_role_registry`
already applies, which forces ``False`` whenever ``feasible`` is ``False``
(and every ``peer_proxied`` name is infeasible by construction).
"""

from __future__ import annotations

import copy
import dataclasses
import json

from lobes.gateway._config import build_config
from lobes.gateway._routing import Backend, RoutingTable
from lobes.roles import ROLE_BACKEND, ROLES, annotate_peer_referrals, build_role_registry

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
_GATEWAY_URL = "http://localhost:8000"

# Peer origins declared by an operator — deliberately "weird but valid" shapes
# (an IPv6 literal, a non-default port, a trailing path segment) to prove
# annotate_peer_referrals never mangles what it was handed (verbatim, #92).
_IPV6_ORIGIN = "http://[::1]:8443"
_TRAILING_PATH_ORIGIN = "http://box.example:8001/mesh/v1"
_THOR_ORIGIN = "http://thor.local:8001"


def _table(
    *,
    infeasible: tuple[str, ...] = (),
    peer_origins: dict[str, str] | None = None,
    peer_proxied: tuple[str, ...] = (),
) -> RoutingTable:
    """A minimal RoutingTable — constructed directly (not via env parsing) so
    these tests exercise ONLY annotate_peer_referrals's own logic, isolated
    from lobes.gateway._config's env-parsing/trimming behaviour (already
    covered by tests/test_gateway_config_proxy.py and tests/test_peer_referral.py)."""
    backend = Backend(name="primary", base_url="http://vllm-primary:8000", served_name=_CORTEX_ID)
    return RoutingTable(
        backends=(backend,),
        default_model=_CORTEX_ID,
        aliases={},
        infeasible=frozenset(infeasible),
        peer_origins=peer_origins or {},
        peer_proxied=frozenset(peer_proxied),
    )


def _spark_lobe_env(*, proxy: bool = False, peers: bool = True, **over) -> dict[str, str]:
    """A rendered spark-lobe-like env: cortex + pooling hosted, senses DROPPED,
    optionally referred-to and/or proxied to a declared peer (Thor)."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "MULTIMODAL_FEASIBLE": "false",
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
    }
    if peers:
        env["MULTIMODAL_PEER_ORIGIN"] = _THOR_ORIGIN
    if proxy:
        env["MULTIMODAL_PEER_PROXY"] = "true"
    env.update(over)
    return env


def _pre_proxy_annotate(payload: dict[str, dict], table: RoutingTable) -> dict[str, dict]:
    """A frozen re-implementation of the PRE-t5 annotate_peer_referrals — the
    referral-only behaviour (hosted_by only, never a proxied marker) that
    shipped before this task. Used as an independent oracle to prove the
    byte-identical guarantee, rather than trusting the very function under
    test to also prove its own unchanged-ness."""
    for role, backend in ROLE_BACKEND.items():
        entry = payload.get(role)
        if not isinstance(entry, dict) or entry.get("feasible") is not False:
            continue
        origin = table.peer_origins.get(backend)
        if origin:
            entry["hosted_by"] = origin
    return payload


# ============================================================================
# Unit level: annotate_peer_referrals directly, hand-built payloads
# ============================================================================


def test_proxied_role_gets_proxied_marker_and_hosted_by_verbatim() -> None:
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": _THOR_ORIGIN},
        peer_proxied=("multimodal",),
    )
    payload = {"cortex": {"feasible": True}, "senses": {"feasible": False}}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["proxied"] is True
    assert payload["senses"]["hosted_by"] == _THOR_ORIGIN
    assert payload["senses"]["feasible"] is False


def test_referral_only_role_has_hosted_by_no_proxied_marker() -> None:
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": _THOR_ORIGIN},
        peer_proxied=(),  # NOT opted into proxying — referral only
    )
    payload = {"senses": {"feasible": False}}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["hosted_by"] == _THOR_ORIGIN
    assert "proxied" not in payload["senses"]


def test_hosted_role_unchanged_shape_even_with_proxying_declared_elsewhere() -> None:
    # A hosted role (feasible=True) must never gain hosted_by/proxied, even
    # when OTHER roles in the same table are proxied.
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": _THOR_ORIGIN},
        peer_proxied=("multimodal",),
    )
    payload = {"cortex": {"feasible": True}, "senses": {"feasible": False}}
    before = copy.deepcopy(payload["cortex"])
    annotate_peer_referrals(payload, table)
    assert payload["cortex"] == before
    assert "hosted_by" not in payload["cortex"]
    assert "proxied" not in payload["cortex"]


def test_dropped_role_with_no_declared_peer_gets_neither_key() -> None:
    # infeasible but nothing declared at all: no hosted_by, no proxied,
    # regardless of what peer_proxied says elsewhere (unreachable code path in
    # practice — _peer_proxied requires a declared origin — but the annotator
    # itself must not assume that invariant blindly).
    table = _table(infeasible=("rerank",), peer_origins={}, peer_proxied=())
    payload = {"reranker": {"feasible": False}}
    annotate_peer_referrals(payload, table)
    assert payload["reranker"] == {"feasible": False}


def test_three_states_distinguished_by_key_presence_in_one_table() -> None:
    """hosted / referral-only / proxied, side by side in a single table —
    the three states are told apart by key PRESENCE, never a sentinel value."""
    table = _table(
        infeasible=("multimodal", "embed", "rerank"),
        peer_origins={"multimodal": _THOR_ORIGIN, "embed": _TRAILING_PATH_ORIGIN},
        peer_proxied=("multimodal",),
    )
    payload = {
        "cortex": {"feasible": True},  # hosted
        "senses": {"feasible": False},  # proxied
        "embedder": {"feasible": False},  # referral-only
        "reranker": {"feasible": False},  # dropped, no peer at all
    }
    annotate_peer_referrals(payload, table)

    # hosted: neither key.
    assert set(payload["cortex"]) == {"feasible"}
    # proxied: both keys, proxied is a real bool True (not a string/int/etc).
    assert payload["senses"]["hosted_by"] == _THOR_ORIGIN
    assert payload["senses"]["proxied"] is True
    assert set(payload["senses"]) == {"feasible", "hosted_by", "proxied"}
    # referral-only: hosted_by present, proxied ABSENT (never False).
    assert payload["embedder"]["hosted_by"] == _TRAILING_PATH_ORIGIN
    assert "proxied" not in payload["embedder"]
    assert set(payload["embedder"]) == {"feasible", "hosted_by"}
    # dropped with no declared peer: untouched.
    assert set(payload["reranker"]) == {"feasible"}


# ============================================================================
# Origin round-trips verbatim — weird-but-valid shapes are never mangled
# ============================================================================


def test_origin_round_trips_verbatim_ipv6() -> None:
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": _IPV6_ORIGIN},
        peer_proxied=("multimodal",),
    )
    payload = {"senses": {"feasible": False}}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["hosted_by"] == _IPV6_ORIGIN
    assert payload["senses"]["proxied"] is True


def test_origin_round_trips_verbatim_trailing_path() -> None:
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": _TRAILING_PATH_ORIGIN},
        peer_proxied=("multimodal",),
    )
    payload = {"senses": {"feasible": False}}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["hosted_by"] == _TRAILING_PATH_ORIGIN


def test_origin_never_derived_normalized_when_proxied() -> None:
    # The #92 lesson still holds when proxied: the origin the annotator writes
    # is EXACTLY what the table carried — not re-derived, not host/port
    # recombined, not scheme-normalized.
    weird = "https://Weird-CASE.example:9443/a/b/../c"
    table = _table(
        infeasible=("multimodal",),
        peer_origins={"multimodal": weird},
        peer_proxied=("multimodal",),
    )
    payload = {"senses": {"feasible": False}}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["hosted_by"] == weird


# ============================================================================
# ready is never hardcoded true for a proxied role
# ============================================================================


def test_proxied_role_ready_is_false_via_build_role_registry() -> None:
    table, cfg = build_config(_spark_lobe_env(proxy=True))
    registry = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL)
    assert registry["senses"].ready is False
    assert registry["senses"].loaded is False  # spark-lobe drops the container too


def test_proxied_role_ready_stays_false_even_with_a_stray_live_true_signal() -> None:
    # A caller-supplied backend_ready=True for the dropped backend must not
    # resurrect ready=True for a proxied role: feasible=False clamps it,
    # exactly as it already clamps a referral-only dropped role. Nothing here
    # invents a "proxied, therefore trust the signal" exception.
    table, cfg = build_config(_spark_lobe_env(proxy=True))
    registry = build_role_registry(
        table, cfg, gateway_url=_GATEWAY_URL, backend_ready={"multimodal": True}
    )
    assert registry["senses"].ready is False
    assert registry["senses"].feasible is False


def test_proxied_capabilities_payload_never_claims_ready() -> None:
    table, cfg = build_config(_spark_lobe_env(proxy=True))
    registry = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL)
    payload = {role: dataclasses.asdict(registry[role]) for role in ROLES}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["proxied"] is True
    assert payload["senses"]["ready"] is False


# ============================================================================
# Byte-identical guarantee: peer_proxied empty (the default) == today
# ============================================================================


def test_no_peer_config_payload_matches_pre_proxy_oracle() -> None:
    """No peer config at all — build the payload two ways (the real function
    under test, and the frozen pre-t5 oracle) and assert equality."""
    table, cfg = build_config(_spark_lobe_env(peers=False, proxy=False))
    registry = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL)
    base = {role: dataclasses.asdict(registry[role]) for role in ROLES}

    via_new = annotate_peer_referrals(copy.deepcopy(base), table)
    via_oracle = _pre_proxy_annotate(copy.deepcopy(base), table)
    assert json.dumps(via_new, sort_keys=True) == json.dumps(via_oracle, sort_keys=True)
    assert "hosted_by" not in json.dumps(via_new)
    assert "proxied" not in json.dumps(via_new)


def test_referral_only_payload_matches_pre_proxy_oracle() -> None:
    """A peer origin declared but PEER_PROXY not armed — the exact issue #112
    referral-only contract must render identically to the pre-t5 oracle (the
    proxied branch never fires when peer_proxied is empty)."""
    table, cfg = build_config(_spark_lobe_env(peers=True, proxy=False))
    assert table.peer_proxied == frozenset()  # sanity: the knob really is off
    registry = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL)
    base = {role: dataclasses.asdict(registry[role]) for role in ROLES}

    via_new = annotate_peer_referrals(copy.deepcopy(base), table)
    via_oracle = _pre_proxy_annotate(copy.deepcopy(base), table)
    assert json.dumps(via_new, sort_keys=True) == json.dumps(via_oracle, sort_keys=True)
    assert via_new["senses"]["hosted_by"] == _THOR_ORIGIN
    assert "proxied" not in via_new["senses"]


def test_empty_peer_proxied_is_the_default_on_a_bare_table() -> None:
    # A RoutingTable built with no explicit peer_proxied argument defaults to
    # empty, exactly like peer_origins/infeasible — an untouched construction
    # (this module's own tests, and every other caller's) is unaffected.
    backend = Backend(name="primary", base_url="http://x:8000", served_name=_CORTEX_ID)
    table = RoutingTable(backends=(backend,), default_model=_CORTEX_ID, aliases={})
    assert table.peer_proxied == frozenset()


# ============================================================================
# Full-fleet sanity: only the proxied role carries the marker
# ============================================================================


def test_only_the_proxied_backend_role_carries_the_marker_in_full_payload() -> None:
    table, cfg = build_config(_spark_lobe_env(proxy=True))
    registry = build_role_registry(table, cfg, gateway_url=_GATEWAY_URL)
    payload = {role: dataclasses.asdict(registry[role]) for role in ROLES}
    annotate_peer_referrals(payload, table)
    assert payload["senses"]["proxied"] is True
    for role in ("cortex", "embedder", "reranker", "stt", "tts"):
        assert "proxied" not in payload[role], role
        assert "hosted_by" not in payload[role], role
