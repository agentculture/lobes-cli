"""Decider domains on the advert and over the mesh (orin-decision-models t7).

The decider namespace is FLAT and mesh-wide (c28/c41): a caller names
``decider:<domain>`` in ``X-Lobes-Model`` and whichever member serves that
domain answers. So:

* ``GET /capabilities`` lists the decider with a ``domains`` map (domain ->
  checkpoint, quant, ready), serialised through ``lobes.roles.role_payload``;
* the mesh announcement carries the SAME domains, each under its own roster
  key ``decider:<domain>``, and a peer's verification probe reads them back
  under the same keys;
* a member that serves no decider forwards each domain, in ONE hop, to the
  member announcing it — body verbatim, ``X-Lobes-Mesh-Member`` on the reply;
* two members announcing one domain with DIFFERENT fingerprints are refused
  (409), never silently picked.
"""

from __future__ import annotations

import json

import pytest

from lobes.catalog import SUPPORTED_MODELS
from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_config import MeshConfig
from lobes.gateway._mesh_routes import _probe_member_capabilities, announcement_from_capabilities
from lobes.gateway._mesh_routing import (
    MESH_MEMBER_HEADER,
    MemberInfo,
    RoutingSnapshot,
    decider_domain_placement,
    verify_member_roles,
)
from lobes.gateway._mesh_wire import capability_entries, decode, encode
from lobes.gateway._routing import DECIDER_MODEL_HEADER, SYSTEMONE_PATH
from lobes.roles import (
    build_role_registry,
    decider_domain_contexts,
    decider_domain_models,
    role_payload,
)
from tests.test_mesh_routing_wiring import _FakeUpstream

JOIN_KEY = "sk-mesh-join-key"
ORIN = "http://orin:8000"
SPARK2 = "http://spark2:8000"
THOR = "http://thor:8000"

CLEF = "culture-dev/clef-27b-GPTQ-W4A16-g64"  # the default decider (odm-t22)
STRANDS = "StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"

# Byte-exact, deliberately not what json.dumps would emit: a forward that
# re-serialised the body would change these bytes.
_BODY = b'{"state":  {"n": 1},\n "questions": [{"id": "q1"}]}\n'
_ANSWER = b'{"answers" :[{"id":"q1","value":true}]}  '

_ORIN_ENV = {"DECIDER_DOMAINS": "general=http://decider-general:8000"}
_SPARK2_ENV = {
    "DECIDER_DOMAINS": "triage=http://decider-triage:8000",
    "DECIDER_DOMAIN_MODELS": f"triage={STRANDS}",
}


@pytest.fixture
def mesh_env(monkeypatch):
    monkeypatch.setenv("LOBES_MESH_KEY", JOIN_KEY)
    monkeypatch.setenv("LOBES_MESH_NAME", "thor")


def _mesh_cfg(name: str) -> MeshConfig:
    return MeshConfig(
        enabled=True,
        key=JOIN_KEY,
        name=name,
        seeds=(),
        heartbeat_s=60,
        missed_max=3,
        ledger_path=None,
    )


def _hosting_payload(env: dict[str, str], origin: str) -> dict:
    """A hosting box's own /capabilities: every backend and domain ready."""
    table, cfg = build_config(env)
    ready = {b.name: True for b in table.backends}
    ready.update(dict.fromkeys(S.decider_readiness_targets(table), True))
    return S.capabilities_payload(
        table,
        cfg,
        env,
        gateway_url=origin,
        backend_ready=ready,
        mesh_snapshot=RoutingSnapshot(members=(), announcements=()),
    )


def _announce(env: dict[str, str], name: str, origin: str):
    payload = _hosting_payload(env, origin)
    ann = announcement_from_capabilities(_mesh_cfg(name), payload, self_origin=origin)
    return decode(encode(ann)), payload  # over the wire and back


def _probed(payload: dict) -> dict[str, dict]:
    return {
        key: {"fingerprint": entry.get("fingerprint"), "ready": entry.get("ready")}
        for key, entry in capability_entries(payload).items()
        if isinstance(entry, dict) and not entry.get("proxied")
    }


def _member(name: str, origin: str, env: dict[str, str], *, probed: bool = True):
    ann, payload = _announce(env, name, origin)
    verified = verify_member_roles(ann, _probed(payload)) if probed else frozenset()
    info = MemberInfo(
        name=name,
        origin=origin,
        announced_roles=tuple(ann.roles),
        verified_roles=tuple(sorted(verified)),
        capacity=1.0,
        probed=probed,
        ready_roles=tuple(sorted(verified)),
    )
    return info, ann


def _snapshot(*members) -> RoutingSnapshot:
    return RoutingSnapshot(
        members=tuple(m for m, _a in members),
        announcements=tuple((m.origin, a) for m, a in members),
    )


def _a_and_b() -> RoutingSnapshot:
    return _snapshot(_member("orin", ORIN, _ORIN_ENV), _member("spark2", SPARK2, _SPARK2_ENV))


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, backend, path, fwd_body, headers, *, connect_timeout, read_timeout):
        self.calls.append(
            {"base_url": backend.base_url, "path": path, "body": fwd_body, "headers": headers}
        )
        return _FakeUpstream(200, _ANSWER)


def _systemone(table, cfg, rec, snap, *, domain: str | None, headers=()):
    req = [("Authorization", "Bearer sk-caller"), *headers]
    if domain is not None:
        req.append((DECIDER_MODEL_HEADER, domain))
    return S.handle_systemone(table, cfg, SYSTEMONE_PATH, req, _BODY, rec, mesh_snapshot=snap)


def _hdr(resp, name: str) -> list[str]:
    return [v for k, v in resp.headers if k.lower() == name.lower()]


def _sent(call, name: str) -> list[str]:
    return [v for k, v in call["headers"] if k.lower() == name.lower()]


# --- criterion 1: the advert's domains map ------------------------------------


def test_capabilities_lists_decider_with_a_domains_map():
    env = {
        "DECIDER_DOMAINS": "general=http://decider-general:8000,triage=http://decider-triage:8000",
        "DECIDER_DOMAIN_MODELS": f"triage={STRANDS}",
    }
    table, cfg = build_config(env)
    payload = S.capabilities_payload(
        table,
        cfg,
        env,
        gateway_url=ORIN,
        backend_ready={"decider": True, "decider:general": True, "decider:triage": False},
    )
    domains = payload["decider"]["domains"]
    assert set(domains) == {"general", "triage"}
    assert domains["general"]["checkpoint"] == CLEF
    assert domains["general"]["quant"] == "compressed-tensors"
    assert domains["general"]["ready"] is True
    assert domains["triage"]["checkpoint"] == STRANDS
    assert domains["triage"]["quant"] == "none"
    assert domains["triage"]["ready"] is False  # its own sidecar's signal, not general's


def test_no_decider_domain_means_no_domains_key():
    table, cfg = build_config({})
    payload = S.capabilities_payload(table, cfg, {}, gateway_url=ORIN)
    assert "domains" not in payload["decider"]


def test_domains_go_through_role_payload():
    env = {"DECIDER_DOMAINS": "general=http://g:8000"}
    table, cfg = build_config(env)
    info = build_role_registry(table, cfg, env=env, gateway_url=ORIN)["decider"]
    assert info.domains is not None
    assert role_payload(info)["domains"]["general"]["checkpoint"] == CLEF


def test_an_infeasible_decider_never_advertises_a_ready_domain():
    env = {"DECIDER_DOMAINS": "general=http://g:8000", "DECIDER_FEASIBLE": "false"}
    table, cfg = build_config(env)
    payload = S.capabilities_payload(
        table, cfg, env, gateway_url=ORIN, backend_ready={"decider:general": True}
    )
    assert all(d["ready"] is False for d in payload["decider"].get("domains", {}).values())


def test_domain_models_env_skips_malformed_pairs():
    env = {"DECIDER_DOMAIN_MODELS": f"ops={STRANDS}, bad, =x,Up=y,triage=a/b@rev"}
    assert decider_domain_models(env) == {"ops": STRANDS, "triage": "a/b@rev"}


def test_readiness_targets_are_one_per_domain():
    table, _cfg = build_config({"DECIDER_DOMAINS": "general=http://g:1,triage=http://t:1"})
    assert S.decider_readiness_targets(table) == {
        "decider:general": "http://g:1",
        "decider:triage": "http://t:1",
    }


# --- criterion 1: the announcement carries the same domains --------------------


def test_announcement_carries_each_domain_as_its_own_roster_key():
    ann, payload = _announce(_SPARK2_ENV, "spark2", SPARK2)
    assert "decider:triage" in ann.roles
    info = ann.roles["decider:triage"]
    assert info.model == payload["decider"]["domains"]["triage"]["checkpoint"] == STRANDS
    assert info.quant == payload["decider"]["domains"]["triage"]["quant"]
    assert info.fingerprint.served_id == STRANDS
    # Exactly the domains the advert lists — no more.
    announced = {k.split(":", 1)[1] for k in ann.roles if k.startswith("decider:")}
    assert announced == set(payload["decider"]["domains"])


def test_peer_probe_verifies_announced_domains(monkeypatch):
    ann, payload = _announce(_ORIN_ENV, "orin", ORIN)
    monkeypatch.setattr(
        "lobes.gateway._readiness._default_peer_opener",
        lambda url, timeout, key: (200, json.dumps(payload).encode()),
    )
    origin, verified, ready, reason, _ctx, _models = _probe_member_capabilities(
        ("orin", ORIN, ann), JOIN_KEY, 1.0
    )
    assert origin == ORIN
    assert "decider:general" in verified
    assert "decider:general" in ready
    assert reason is None


# --- criterion 2: route by domain across the mesh ------------------------------


@pytest.mark.parametrize(
    ("domain", "origin", "member"),
    [
        ("decider:general", ORIN, "orin"),
        (None, ORIN, "orin"),  # no header means decider:general
        ("decider:triage", SPARK2, "spark2"),
    ],
)
def test_member_c_forwards_each_domain_to_its_member(mesh_env, domain, origin, member):
    table, cfg = build_config({})  # member C hosts no decider at all
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, _a_and_b(), domain=domain)

    assert resp.status == 200
    assert len(rec.calls) == 1
    call = rec.calls[0]
    assert call["base_url"].rstrip("/") == origin
    assert call["path"] == SYSTEMONE_PATH
    assert call["body"] == _BODY  # byte-for-byte (c45)
    expected = domain or "decider:general"
    assert _sent(call, DECIDER_MODEL_HEADER) == [expected]
    assert _sent(call, "Authorization") == [f"Bearer {JOIN_KEY}"]
    assert _sent(call, S.PROXIED_HEADER)  # single-hop marker
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == member
    assert origin in _hdr(resp, S.PROXIED_BY_HEADER)


def test_two_members_one_domain_different_fingerprints_is_refused(mesh_env):
    clashing = {
        "DECIDER_DOMAINS": "general=http://decider-general:8000",
        "DECIDER_DOMAIN_MODELS": f"general={STRANDS}",
    }
    snap = _snapshot(_member("orin", ORIN, _ORIN_ENV), _member("spark2", SPARK2, clashing))
    assert decider_domain_placement(snap, "general").kind == "conflict"
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, snap, domain="decider:general")
    assert rec.calls == []  # never silently picked
    assert resp.status == 409
    err = json.loads(resp.body)["error"]
    assert err["type"] == "decider_domain_conflict"
    assert err["members"] == ["orin", "spark2"]


def test_identical_fingerprints_are_replicas_of_one_domain(mesh_env):
    snap = _snapshot(_member("spark2", SPARK2, _ORIN_ENV), _member("orin", ORIN, _ORIN_ENV))
    placement = decider_domain_placement(snap, "general")
    assert (placement.kind, placement.member) == ("forward", "orin")  # first by name


def test_a_domain_served_here_is_never_forwarded(mesh_env):
    table, cfg = build_config({"DECIDER_DOMAINS": "general=http://local-decider:8000"})
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, _a_and_b(), domain="decider:general")
    assert resp.status == 200
    assert rec.calls[0]["base_url"] == "http://local-decider:8000"
    assert not _hdr(resp, MESH_MEMBER_HEADER)


def test_a_box_with_other_domains_forwards_the_one_it_lacks(mesh_env):
    table, cfg = build_config({"DECIDER_DOMAINS": "general=http://local-decider:8000"})
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, _a_and_b(), domain="decider:triage")
    assert resp.status == 200
    assert rec.calls[0]["base_url"].rstrip("/") == SPARK2
    assert _hdr(resp, MESH_MEMBER_HEADER)[-1] == "spark2"


def test_hop_marked_arrival_is_never_forwarded_again(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _systemone(
        table, cfg, rec, _a_and_b(), domain="decider:triage", headers=[(S.PROXIED_HEADER, "x")]
    )
    assert rec.calls == []
    assert resp.status == 508


def test_pending_domain_answers_503_role_unverified(mesh_env):
    snap = _snapshot(_member("spark2", SPARK2, _SPARK2_ENV, probed=False))
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, snap, domain="decider:triage")
    assert rec.calls == []
    assert resp.status == 503
    assert json.loads(resp.body)["error"]["hosted_by"] == SPARK2


def test_a_domain_nobody_announces_keeps_the_local_refusal(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, _a_and_b(), domain="decider:nowhere")
    assert rec.calls == []
    assert resp.status == 404
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"


def test_illegal_spelling_never_reaches_the_mesh(mesh_env):
    table, cfg = build_config({})
    rec = _Recorder()
    resp = _systemone(table, cfg, rec, _a_and_b(), domain="cortex")
    assert rec.calls == []
    assert json.loads(resp.body)["error"]["type"] == "model_not_found"


# --- member C's own advert names the mesh-reached domains ----------------------


def test_member_c_capabilities_lists_mesh_domains_as_proxied():
    table, cfg = build_config({})
    payload = S.capabilities_payload(table, cfg, {}, gateway_url=THOR, mesh_snapshot=_a_and_b())
    domains = payload["decider"]["domains"]
    assert domains["general"] == {
        "checkpoint": CLEF,
        "quant": "compressed-tensors",
        "ready": True,
        "proxied": True,
        "hosted_by": ORIN,
    }
    assert domains["triage"]["hosted_by"] == SPARK2
    # ... and member C never re-announces a peer's domain.
    ann = announcement_from_capabilities(_mesh_cfg("thor"), payload, self_origin=THOR)
    assert not [k for k in ann.roles if k.startswith("decider:")]


def test_member_c_capabilities_lists_a_conflict_not_a_pick():
    clashing = {
        "DECIDER_DOMAINS": "general=http://decider-general:8000",
        "DECIDER_DOMAIN_MODELS": f"general={STRANDS}",
    }
    snap = _snapshot(_member("orin", ORIN, _ORIN_ENV), _member("spark2", SPARK2, clashing))
    table, cfg = build_config({})
    payload = S.capabilities_payload(table, cfg, {}, gateway_url=THOR, mesh_snapshot=snap)
    assert payload["decider"]["domains"]["general"] == {
        "conflict": ["orin", "spark2"],
        "ready": False,
    }


# --- the served window reaches the advert (code review, finding 4) -------------


def test_a_declared_domain_context_is_advertised_and_fingerprinted():
    """The shape renders each domain's served window into DECIDER_DOMAIN_CONTEXTS;
    without it the advert carried the catalog's native 262144 against a 16384
    lane."""
    env = {
        "DECIDER_DOMAINS": "general=http://decider-general:8000,triage=http://decider-triage:8000",
        "DECIDER_DOMAIN_MODELS": f"triage={STRANDS}",
        "DECIDER_DOMAIN_CONTEXTS": "general=16384,triage=8192",
    }
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env, gateway_url=ORIN)
    decider = payload["decider"]
    assert decider["context"] == 16384
    assert decider["domains"]["general"]["fingerprint"]["max_model_len"] == 16384
    assert decider["domains"]["triage"]["fingerprint"]["max_model_len"] == 8192


def test_without_domain_contexts_the_advert_is_unchanged():
    env = {"DECIDER_DOMAINS": "general=http://decider-general:8000"}
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env, gateway_url=ORIN)
    native = next(m for m in SUPPORTED_MODELS if m.id == CLEF).native_max_model_len
    assert payload["decider"]["context"] == native
    assert payload["decider"]["domains"]["general"]["fingerprint"]["max_model_len"] == native


def test_domain_contexts_env_skips_malformed_pairs():
    env = {"DECIDER_DOMAIN_CONTEXTS": "general=16384, bad, ops=x, Up=5, triage=0, a=-1,b=2048"}
    assert decider_domain_contexts(env) == {"general": 16384, "b": 2048}
