"""The ``gateway-only`` deployment shape — the consumer-only mesh member (mesh-brain-join t5).

A box that serves NOTHING locally and draws on every lobe in the mesh: the
first built-in shape with ``hosts = []``. The schema always admitted it (no
cardinality check) and the renderer always knew its projection —
``shape_services`` renders exactly ``("gateway",)`` and every dropped core
role flags off as ``<PREFIX>_FEASIBLE=false`` — but no built-in shape shipped
it, and the gateway's routing layer documented the converse assumption:
"a built table always has a primary backend" (``order_backends``'s invariant
comment at ``lobes/gateway/_routing.py``).

This module is the shape's acceptance (mesh-brain-join plan t5):

1. **The render.** On every card: services == ``("gateway",)``; every core
   role ``FEASIBLE=false`` with no model/knob leak; goldens for
   spark/thor/orin/base.
2. **The relaxed routing.** ``build_config`` on this shape's env — which
   wires no local backend — returns a ``RoutingTable`` whose
   ``order_backends`` yields nothing locally and no exception; a table with
   no backends at all (the mesh-routed table t7 will build) behaves the same
   way, and a feasible owner still routes exactly as before.
3. **The grep gate.** None of ``MESH``/``PEER``/``SELF_ORIGIN`` appears in
   ``lobes/profiles/*.py``: the profile package renders what a box HOSTS,
   and membership is a runtime gateway fact, never a rendered key (the spec's
   honesty condition — render.py / shape_render.py / schema.py contain no
   mesh, roster, peer or self-origin logic — in one check).
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import FEASIBLE_ENV, build_config
from lobes.gateway._routing import RoutingTable, order_backends
from lobes.profiles.loader import builtin_names, resolve_profile
from lobes.profiles.render import ROLE_ENV_PREFIX, profile_env
from lobes.profiles.schema import ROLES as CORE_ROLES
from lobes.profiles.shape_render import GATEWAY_SERVICE, shape_env, shape_services
from lobes.profiles.shapes import (
    OPT_IN_CORE_ROLES,
    SHAPE_ROLES,
    builtin_shape_names,
    resolve_shape,
)
from tests.goldens.regen import shape_golden_pairs, shape_golden_path

# The five non-opt-in core roles flag off on EVERY card under this shape
# (the opt-in core roles pass the card's own declaration through instead —
# see test_every_core_role_flags_off_with_no_knob_leak_on_every_card).
_ALWAYS_DROPPED_ROLES: tuple[str, ...] = ("cortex", "senses", "hand", "embedder", "reranker")


# --- the shape's own data ----------------------------------------------------


def test_loads_as_a_builtin_shape_that_hosts_nothing() -> None:
    shape = resolve_shape("gateway-only")
    assert shape.name == "gateway-only"
    # hosts = []: not even `hand`, the cheap role every OTHER built-in shape
    # hosts. A box that serves nothing runs nothing.
    assert shape.hosts == ()
    assert not any(shape.hosts_role(role) for role in SHAPE_ROLES)
    assert "gateway-only" in builtin_shape_names()


def test_carries_no_overrides_because_nothing_is_hosted_to_reclaim() -> None:
    # The lobe shapes' overrides exist to SPEND a dropped lobe's budget; this
    # shape drops every lobe, so there is nothing to re-derive and the card's
    # declarations are simply never rendered.
    assert dict(resolve_shape("gateway-only").overrides) == {}


def test_claims_no_validation() -> None:
    """#108: no box has booted this shape yet — the plan's t12 fourth-member
    test is its first live run — so neither the summary nor the file may
    imply one has."""
    text = (
        files("lobes.profiles.builtin_shapes")
        .joinpath("gateway-only.toml")
        .read_text(encoding="utf-8")
    )
    assert "UNVALIDATED" in resolve_shape("gateway-only").summary
    assert "DECLARED-BUT-UNVALIDATED" in text
    assert "VALIDATED on" not in text


# --- the render: services and flagged-off roles on every card (criterion 1) --


@pytest.mark.parametrize("card", builtin_names())
def test_renders_only_the_gateway_service_on_every_card(card: str) -> None:
    shape = resolve_shape("gateway-only")
    services = shape_services(shape, resolve_profile(card))
    # Exactly ('gateway',): no role service, no audio overlay, no realtime
    # bridge, no opt-in gear — a box that serves nothing runs nothing but the
    # fleet front.
    assert services == (GATEWAY_SERVICE,)


@pytest.mark.parametrize("card", builtin_names())
def test_every_core_role_flags_off_with_no_knob_leak_on_every_card(card: str) -> None:
    shape = resolve_shape("gateway-only")
    profile = resolve_profile(card)
    env = shape_env(shape, profile)
    for role in CORE_ROLES:
        prefix = ROLE_ENV_PREFIX[role]
        # innereye's `declared_peak_gib` (t6, issue #268) is a card-level FACT,
        # not a serving decision — it passes through on every shape exactly
        # like `host_env` does, hosted or not, because it is what the
        # co-residency veto reads regardless of which shape is being
        # considered. It is the one legitimate exception to "dropped role,
        # no knob leak" below.
        allowed_stray = f"{prefix}_DECLARED_PEAK_GIB" if role == "innereye" else None
        if role in OPT_IN_CORE_ROLES:
            # Non-hosted opt-in core roles pass the card's own declaration
            # through: the base card's veto renders its marker, a silent card
            # renders NOTHING (the gateway's OPT_IN_BACKENDS unwired default
            # carries the infeasibility) — the convention test_shape_goldens
            # pins for every shape x card. A card that is FEASIBLE but only
            # carries a card-level fact (no marker at all) is the third case.
            if role in profile.roles and not profile.role(role).feasible:
                expected = "false"
            else:
                expected = None
        else:
            expected = "false"
        assert env.get(f"{prefix}_FEASIBLE") == expected, role
        stray = [
            k
            for k in env
            if k.startswith(f"{prefix}_") and k != f"{prefix}_FEASIBLE" and k != allowed_stray
        ]
        assert stray == [], f"dropped {role} leaked {stray}"


@pytest.mark.parametrize("card", builtin_names())
def test_renders_no_local_wiring_and_no_compose_profile_on_every_card(card: str) -> None:
    shape = resolve_shape("gateway-only")
    env = shape_env(shape, resolve_profile(card))
    wiring = [k for k in env if k.endswith(("_URL", "_BASE_URL", "_SERVED_NAME", "_MODEL"))]
    assert wiring == []
    assert "COMPOSE_PROFILES" not in env
    # Everything rendered is either a flagged-off marker or the card's own
    # box-level passthrough (host_env) — nothing else.
    bare_card = set(profile_env(resolve_profile(card)))
    markers = {f"{ROLE_ENV_PREFIX[role]}_FEASIBLE" for role in CORE_ROLES}
    for key in env:
        if key not in markers:
            assert key in bare_card, f"{card}: unexpected key {key!r}"


# --- the goldens for spark/thor/orin/base (criterion 1) ----------------------


def test_gateway_only_gets_a_golden_for_every_card() -> None:
    # The discovery mechanism picks the new shape up with zero edits: hosts=[]
    # diverges from the bare card profile, so it is NOT the identity shape and
    # earns its own per-card goldens.
    cards = {card for shape, card in shape_golden_pairs() if shape == "gateway-only"}
    assert cards == set(builtin_names())


@pytest.mark.parametrize("card", builtin_names())
def test_goldens_exist_for_spark_thor_orin_base(card: str) -> None:
    path = shape_golden_path("gateway-only", card)
    assert (
        path.is_file()
    ), f"missing golden {path} — regenerate: uv run python tests/goldens/regen.py"


@pytest.mark.parametrize("card", builtin_names())
def test_golden_carries_only_flagged_off_markers_and_card_passthrough(card: str) -> None:
    golden = shape_golden_path("gateway-only", card).read_text(encoding="utf-8")
    for role in CORE_ROLES:
        prefix = ROLE_ENV_PREFIX[role]
        # innereye's `declared_peak_gib` (t6, issue #268) is the one
        # legitimate non-marker line for a dropped role — a card-level fact
        # the co-residency veto reads, passed through regardless of hosting
        # (see test_every_core_role_flags_off_with_no_knob_leak_on_every_card).
        allowed = f"{prefix}_DECLARED_PEAK_GIB=31.42" if role == "innereye" else None
        for line in golden.splitlines():
            if line.startswith(f"{prefix}_"):
                assert line in (f"{prefix}_FEASIBLE=false", allowed), f"unexpected line {line!r}"
    # The five non-opt-in core roles flag off on EVERY card.
    for role in _ALWAYS_DROPPED_ROLES:
        assert f"{ROLE_ENV_PREFIX[role]}_FEASIBLE=false" in golden


# --- the relaxed routing invariant (criterion 2) ------------------------------


def _gateway_only_env(card: str) -> dict[str, str]:
    """The .env a gateway-only box carries for the gateway: the shape's own
    render (flagged-off markers + card passthrough) — no ``*_URL`` /
    ``*_BASE_URL`` / ``*_SERVED_NAME`` wiring anywhere, i.e. an env with no
    local backend."""
    env = shape_env(resolve_shape("gateway-only"), resolve_profile(card))
    assert not [k for k in env if k.endswith(("_URL", "_BASE_URL", "_SERVED_NAME"))]
    return env


@pytest.mark.parametrize("card", builtin_names())
def test_build_config_on_a_no_local_backend_env_builds_without_exception(card: str) -> None:
    env = _gateway_only_env(card)
    table, cfg = build_config(env)  # no exception — the acceptance
    assert cfg is not None
    # The only entry is the synthesized default primary — the vllm-primary
    # container this box does not run — and every core role is infeasible, so
    # nothing local can actually serve.
    assert [b.name for b in table.backends] == ["primary"]
    assert "primary" in table.infeasible
    assert table.infeasible == frozenset(
        name for name in FEASIBLE_ENV if name not in ("stt", "tts")
    )


@pytest.mark.parametrize("card", builtin_names())
def test_order_backends_yields_nothing_locally_on_the_gateway_only_table(card: str) -> None:
    table, _ = build_config(_gateway_only_env(card))
    # The relaxed invariant (the old one claimed "a built table always has a
    # primary backend" and owner non-None in practice): a box that declares
    # every core role infeasible owns NOTHING locally — the default model,
    # every tier/role alias, and a raw served name alike yield no local
    # backend, and no exception escapes.
    assert order_backends(table, table.default_model) == []
    for alias in sorted(table.aliases):
        assert order_backends(table, alias) == [], alias
    for name in ("cortex", "senses", "hand", "muse", "worker", "associate"):
        assert order_backends(table, name) == []
    assert order_backends(table, "a/never-hosted-model") == []


def test_a_table_with_no_backends_at_all_builds_and_yields_nothing() -> None:
    # The mesh-routed table (t7) will carry the mesh's lanes instead of local
    # ones — backends=() outright. It builds, order_backends yields nothing
    # locally, and no exception escapes: the mesh layer owns the request.
    table = RoutingTable(backends=(), default_model="x/whatever", aliases={})
    assert order_backends(table, "x/whatever") == []
    assert order_backends(table, "anything") == []


def test_a_feasible_owner_still_routes_locally() -> None:
    # The relaxation only removes INFEASIBLE owners: a table whose owner is
    # feasible (every pre-mesh deployment) routes exactly as before.
    table, _ = build_config(
        {"PRIMARY_URL": "http://vllm-primary:8000", "PRIMARY_SERVED_NAME": "p/id"}
    )
    assert [b.name for b in order_backends(table, "p/id")] == ["primary"]
    assert [b.name for b in order_backends(table, "main")] == ["primary"]
    # And the general form of the relaxation: WIRED-but-infeasible owns
    # nothing local either (thor-lobe's dropped cortex is exactly this shape,
    # where the infeasibility gate 404s the request before ordering).
    mixed, _ = build_config(
        {
            "PRIMARY_URL": "http://vllm-primary:8000",
            "PRIMARY_SERVED_NAME": "p/id",
            "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
            "MULTIMODAL_SERVED_NAME": "m/id",
            "MULTIMODAL_FEASIBLE": "false",
        }
    )
    assert order_backends(mixed, "m/id") == []
    assert [b.name for b in order_backends(mixed, "p/id")] == ["primary"]


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


def _opener():
    """An ``open_upstream`` stub recording every backend it is asked to dial."""
    calls: list[str] = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append(backend.name)
        return _FakeUpstream()

    return opener, calls


def test_unspecified_model_404s_the_infeasible_default_honestly() -> None:
    # The matrix's orin-small contract, at its extreme: the default model is
    # the (dropped) cortex id, so a request with NO model field 404s
    # role_infeasible — never a dial to a lane the box declared it cannot run.
    table, cfg = build_config(_gateway_only_env("spark"))
    opener, calls = _opener()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b"{}", opener)
    assert resp.status == 404
    assert calls == []
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"


def test_dropped_role_404s_role_infeasible_and_refers_the_declared_peer() -> None:
    # Until t7's auto-wired mesh proxy replaces the hand-typed referral, the
    # dropped role's honest answer is the 404 with hosted_by — nothing local
    # is ever dialed, and the mesh layer (t7) is what follows the referral.
    # Retired (t14): PRIMARY_PEER_ORIGIN no longer populates table.peer_origins
    # — set it directly instead.
    import dataclasses

    env = _gateway_only_env("spark")
    table, cfg = build_config(env)
    table = dataclasses.replace(table, peer_origins={"primary": "http://cortex-peer.local:8001"})
    opener, calls = _opener()
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], json.dumps({"model": "cortex"}).encode(), opener
    )
    assert resp.status == 404
    assert calls == []
    body = json.loads(resp.body)["error"]
    assert body["type"] == "role_infeasible"
    assert body["hosted_by"] == "http://cortex-peer.local:8001"


# --- the grep gate (criterion 3) ---------------------------------------------

_FORBIDDEN_PROFILE_TOKENS = ("MESH", "PEER", "SELF_ORIGIN")


def test_no_mesh_peer_or_self_origin_symbol_in_the_profiles_package() -> None:
    """The grep gate: none of MESH/PEER/SELF_ORIGIN appears in
    ``lobes/profiles/*.py``.

    Membership, referral and self-origin are RUNTIME gateway facts
    (mesh-brain-join t1-t9) — never rendered ``.env`` keys — so the profile
    package must not even name them: #92's operator-typed-origins rule and the
    spec's honesty condition ("render.py, shape_render.py and schema.py
    contain no mesh, roster, peer or self-origin logic") in one check.
    Uppercase tokens on purpose: lowercase "mesh"/"peer" prose about the
    deployment topology is documentation; the symbols are the env-var/constant
    spellings a renderer would have to know to leak membership into a golden.
    """
    profiles_dir = Path(__file__).resolve().parents[1] / "lobes" / "profiles"
    py_files = sorted(profiles_dir.glob("*.py"))
    assert py_files, "the gate proved nothing: no lobes/profiles/*.py found"
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for token in _FORBIDDEN_PROFILE_TOKENS:
            assert token not in text, (
                f"lobes/profiles/{path.name} contains {token!r} — the profile "
                "package renders what a box hosts; membership is a gateway "
                "runtime fact, never a rendered key"
            )
