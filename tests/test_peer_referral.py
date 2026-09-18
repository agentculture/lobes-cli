"""Honest referral: the ``role_infeasible`` 404's ``hosted_by`` (mesh-brain
t3, issue #112) — RETIRED env source, MESH source added (t14).

A box drops a role. Before t14 the only way its honesty surfaces could name
the peer that hosts that role was an operator-declared
``<PREFIX>_PEER_ORIGIN`` env var
(:data:`lobes.gateway._config.PEER_ORIGIN_ENV`, now DELETED). t13 made the
mesh ``RoutingSnapshot`` the candidate source for pool placement and
forwarding; this task extends the SAME mesh snapshot to the referral 404:
:func:`lobes.gateway.server._feasibility_response` (and its audio-lane
sibling in :func:`~lobes.gateway.server.handle_audio_request`) now resolve
``hosted_by`` via :func:`lobes.gateway.server._mesh_referral_origin` — the
origin of the first mesh member that has VERIFIED the role — falling back to
``None`` (the pre-referral body, byte for byte) when no mesh, or no verified
member, is present.

``table.peer_origins`` (the retired env family's field) is asserted
byte-identical to empty regardless of what an operator still has set in
``.env`` — the retired knob is now silently inert, exactly like ``lobes
doctor``'s ``peer_family_retired`` finding describes.

``/capabilities``' own ``hosted_by`` annotation was a recorded GAP here
until t4: :func:`lobes.roles.annotate_peer_referrals` read only the retired
``table.peer_origins``, so the capabilities surface never carried a
mesh-sourced referral while the 404 body did. t4 closed it in
:func:`lobes.roles.annotate_mesh_naming` — which runs LAST, so a
mesh-sourced ``hosted_by`` wins over anything the (still-present,
cite-don't-delete) env annotator would have written. The two surfaces now
agree.

Two invariants remain, now proven via a fake mesh member instead of an env
var:

* **Byte-identity with no mesh and no peer config** — the pre-t3 contract.
* **NO data-plane proxying from a referral-only 404** — a request for an
  unhosted, non-proxied role is answered locally, zero outbound
  connections, whether or not a mesh names a host for it.
"""

from __future__ import annotations

import json

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_routing import build_snapshot
from lobes.gateway._mesh_wire import Fingerprint, RoleInfo
from lobes.gateway._routing import list_models_payload
from lobes.roles import role_payload  # noqa: E402,I001 - the shared advert serializer (d3)
from lobes.roles import ROLES, annotate_peer_referrals, build_role_registry

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
_GATEWAY_URL = "http://localhost:8000"

# A referral origin an operator's peer box would present — never derived (#92).
_THOR_ORIGIN = "http://thor.local:8001"


def _spark_lobe_env(**over) -> dict[str, str]:
    """A rendered spark-lobe env: cortex + pooling hosted, senses DROPPED."""
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _CORTEX_ID,
        "PRIMARY_MAX_MODEL_LEN": "131072",
        "MULTIMODAL_FEASIBLE": "false",
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
    }
    env.update(over)
    return env


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


def _post(table, cfg, model: str, path: str = "/v1/chat/completions", mesh_snapshot=None):
    opener, calls = _opener()
    resp = S.handle_post(
        table,
        cfg,
        path,
        [],
        json.dumps({"model": model}).encode(),
        opener,
        mesh_snapshot=mesh_snapshot,
    )
    return resp, calls


def _mesh_snapshot_for(
    role: str, origin: str = _THOR_ORIGIN, name: str = "thor", *, ready: bool = False
):
    """A minimal mesh RoutingSnapshot with one member verified for *role*.

    ``ready`` seeds ``MemberInfo.ready_roles`` (what the member's probed
    ``/capabilities`` entry reported) — the source of a proxied entry's own
    ``ready`` on this box's ``/capabilities`` (t4). Default ``False`` keeps
    every pre-t4 caller of this helper unchanged.
    """
    fp = Fingerprint(
        served_id=_SENSES_ID, quantization="NVFP4A16", max_model_len=32768, runtime="vllm"
    )

    class _FakeRoster:
        def members(self):
            return [name]

        @property
        def _roster(self):
            rec = type("Rec", (), {"name": name, "origin": origin, "capacity": 1.0})()
            return {name: rec}

        def now(self):
            return 1.0

    return build_snapshot(
        _FakeRoster(),
        announcements={
            origin: type(
                "Ann",
                (),
                {
                    "name": name,
                    "origin": origin,
                    "schema_version": "1.0.0",
                    "roles": {
                        role: RoleInfo(
                            model=_SENSES_ID,
                            runtime="vllm",
                            context=32768,
                            quant="NVFP4A16",
                            responsibilities=("generate",),
                            forbidden_responsibilities=(),
                            fingerprint=fp,
                        )
                    },
                },
            )(),
        },
        verified_roles={origin: frozenset([role])},
        ready_roles={origin: frozenset([role])} if ready else None,
    )


# ============================================================================
# Retired env source: PEER_ORIGIN_ENV is gone, and the knob is now inert
# ============================================================================


def test_peer_origin_env_knob_is_now_inert() -> None:
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN))
    assert dict(table.peer_origins) == {}


def test_peer_origin_is_never_derived_and_never_env_sourced() -> None:
    # The #92 lesson, restated post-retirement: no declaration (env is
    # inert), no mesh member verified for the role => no origin, ever
    # inferred from hostnames/interfaces.
    table, _cfg = build_config(_spark_lobe_env())
    assert "multimodal" in table.infeasible
    assert table.peer_origins.get("multimodal") is None


# ============================================================================
# The mesh source: a verified member's origin becomes `hosted_by`
# ============================================================================


def test_mesh_referral_origin_resolves_a_verified_members_origin() -> None:
    # Unit-level: a verified mesh member's origin is what _feasibility_response
    # would name in `hosted_by`. (End to end, a verified member is actually
    # PLACED by the peer-only pool — see test_mesh_verified_member_is_placed_
    # not_referred below — so this proves the mechanism this task added
    # without the pool's own forward machinery intervening.)
    snap = _mesh_snapshot_for("senses")
    assert S._mesh_referral_origin(snap, "multimodal") == _THOR_ORIGIN


def test_feasibility_response_names_the_mesh_referral() -> None:
    table, _cfg = build_config(_spark_lobe_env())
    snap = _mesh_snapshot_for("senses")
    resp = S._feasibility_response(table, "senses", snap)
    assert resp is not None
    assert resp.status == 404
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _THOR_ORIGIN
    assert _THOR_ORIGIN in body["error"]["message"]


def test_mesh_verified_member_is_placed_not_referred() -> None:
    # End to end: once the mesh VERIFIES a member for the dropped role, the
    # peer-only pool (t13) places the request there instead of falling back
    # to a referral-only 404 — a better outcome than a referral, and it runs
    # BEFORE _feasibility_response is ever reached for this role.
    table, cfg = build_config(_spark_lobe_env())
    snap = _mesh_snapshot_for("senses")
    resp, calls = _post(table, cfg, "senses", mesh_snapshot=snap)
    assert resp.status == 200
    assert calls == ["peer:multimodal"]


def test_404_has_no_referral_when_mesh_has_no_verified_member() -> None:
    table, cfg = build_config(_spark_lobe_env())
    resp, calls = _post(table, cfg, "senses", mesh_snapshot=None)
    assert resp.status == 404
    assert calls == []
    body = json.loads(resp.body)
    assert "hosted_by" not in body["error"]


def test_mesh_referral_for_a_hosted_role_is_never_consulted() -> None:
    # A referral says who hosts a role THIS box does not serve. cortex is
    # hosted here, so infeasible_owner returns None and _mesh_referral_origin
    # is never reached — no 404 is ever built for it, mesh member or not.
    table, _cfg = build_config(_spark_lobe_env())
    resp = S._feasibility_response(table, "cortex", _mesh_snapshot_for("cortex"))
    assert resp is None


def test_embed_mesh_referral_origin_resolves_correctly() -> None:
    env = _spark_lobe_env(EMBED_FEASIBLE="false")
    table, _cfg = build_config(env)
    snap = _mesh_snapshot_for("embedder", origin=_THOR_ORIGIN)
    resp = S._feasibility_response(table, _EMBED_ID, snap)
    assert resp is not None
    assert resp.status == 404
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _THOR_ORIGIN


# ============================================================================
# /capabilities' own hosted_by annotation: the retired env knob is inert, and
# the MESH is now its source (t4). The recorded gap this section used to pin
# — "/capabilities never carries a mesh-sourced referral" — is CLOSED.
# ============================================================================


def test_capabilities_hosted_by_is_never_env_sourced_and_the_env_knob_is_inert() -> None:
    # The retired `<PREFIX>_PEER_ORIGIN` knob: still set in .env, still inert.
    # Without a mesh snapshot nothing names a host for the dropped role.
    env = _spark_lobe_env(MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN)
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert "hosted_by" not in payload["senses"]
    assert "proxied" not in payload["senses"]
    assert "members" not in payload["senses"]
    assert payload["senses"]["feasible"] is False
    assert payload["senses"]["ready"] is False


def test_capabilities_hosted_by_is_mesh_sourced_for_a_dropped_role() -> None:
    # Same inert env knob, now WITH a mesh member verified + ready for the
    # dropped role: hosted_by comes from the mesh member's announced origin,
    # never from the env knob (they are deliberately different strings here).
    env = _spark_lobe_env(MULTIMODAL_PEER_ORIGIN="http://never-used.invalid:9")
    table, cfg = build_config(env)
    snap = _mesh_snapshot_for("senses", ready=True)
    payload = S.capabilities_payload(
        table, cfg, env=env, gateway_url=_GATEWAY_URL, mesh_snapshot=snap
    )
    senses = payload["senses"]
    assert senses["hosted_by"] == _THOR_ORIGIN
    assert senses["proxied"] is True
    assert senses["ready"] is True
    assert senses["member"] == "thor"
    assert "members" not in senses
    # feasible stays a hardware fact — a forward never makes the box host it.
    assert senses["feasible"] is False


def test_capabilities_mesh_annotation_never_touches_a_locally_hosted_role() -> None:
    env = _spark_lobe_env()
    table, cfg = build_config(env)
    snap = _mesh_snapshot_for("senses", ready=True)
    payload = S.capabilities_payload(
        table, cfg, env=env, gateway_url=_GATEWAY_URL, mesh_snapshot=snap
    )
    for role in ("cortex", "embedder", "reranker"):
        assert "hosted_by" not in payload[role], role
        assert "members" not in payload[role], role
        assert "member" not in payload[role], role


def test_annotate_peer_referrals_stays_a_no_op_with_an_always_empty_table() -> None:
    env = _spark_lobe_env()
    table, cfg = build_config(env)
    registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    payload = {role: role_payload(registry[role]) for role in ROLES}
    annotate_peer_referrals(payload, table)
    for role in ROLES:
        assert "hosted_by" not in payload[role], role


# ============================================================================
# Byte-identity regression: zero mesh, zero peer config == the pre-change
# contract
# ============================================================================


def test_capabilities_bytes_identical_without_mesh_or_peer_config() -> None:
    env = _spark_lobe_env()
    table, cfg = build_config(env)
    registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    expected = json.dumps({role: role_payload(registry[role]) for role in ROLES})
    got = json.dumps(S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL))
    assert got == expected
    assert "hosted_by" not in got


def test_role_infeasible_404_bytes_identical_without_mesh_or_peer_config() -> None:
    table, cfg = build_config(_spark_lobe_env())
    resp, calls = _post(table, cfg, "senses")
    assert resp.status == 404
    assert calls == []
    expected = json.dumps(
        {
            "error": {
                "message": (
                    "The model `senses` is not feasible on this machine — its "
                    "backend (`multimodal`) is declared hardware-infeasible "
                    "by this deployment's per-machine profile and will never be "
                    "served here."
                ),
                "type": "role_infeasible",
                "code": "role_infeasible",
            }
        }
    ).encode("utf-8")
    assert resp.body == expected


def test_v1_models_unaffected_by_mesh_referral() -> None:
    # /v1/models stays unchanged either way: it omits the unhosted role and
    # never carries a referral.
    table, _cfg = build_config(_spark_lobe_env())
    ready = {"primary": True, "embed": True, "rerank": True}
    ids = {e["id"] for e in list_models_payload(table, ready)["data"]}
    assert _SENSES_ID not in ids
