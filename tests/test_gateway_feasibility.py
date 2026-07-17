"""Hardware feasibility, end to end (issue #92 extended to the HARDWARE
dimension — plan "per-machine profiles", task t6).

A per-machine :class:`~lobes.profiles.schema.RoleProfile` can declare a role
``feasible=False`` (the box cannot serve it at all, e.g. not enough VRAM for
both cortex and senses at once). This module proves that fact, once it lands
in the gateway container's environment as ``<PREFIX>_FEASIBLE=false`` (the
SAME per-backend-name env convention the served-context overlay already uses
— ``PRIMARY_FEASIBLE`` / ``MULTIMODAL_FEASIBLE`` / ``EMBED_FEASIBLE`` /
``RERANK_FEASIBLE``, see :data:`lobes.gateway._config.FEASIBLE_ENV`), is
honoured EVERYWHERE the existing #92 "advertised implies reachable" invariant
already applies:

* the routing table itself (:mod:`lobes.gateway._routing`) never lets an
  infeasible backend win a request, even via a tier alias's upward-fallback
  substitution or the default-model fallback;
* ``GET /v1/models`` never lists an infeasible backend, even when the live
  readiness cache reports it healthy;
* ``GET /capabilities`` (and, transitively, ``lobes capabilities``, which is
  a client of it — see ``lobes.cli._commands.capabilities``) never reports an
  infeasible role ``ready``, even when a live ``backend_ready`` signal says
  ``True``;
* a POST addressed to an infeasible role (by its role-identity alias, a
  capability-tier alias, or its concrete served model id) gets a 4xx — never
  silently served by a different, feasible gear.

No parallel machinery is invented: :data:`RoutingTable.infeasible` is
computed once, in :func:`lobes.gateway._config.build_config` (the same place
every other env-derived routing fact is computed), and every consumer below
(:func:`lobes.gateway._routing.infeasible_owner`,
:func:`lobes.gateway._routing.list_models_payload`,
:func:`lobes.roles.build_role_registry`) reads it straight off the table
that was already being threaded through — no new parameter on any existing
public builder.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import FEASIBLE_ENV, build_config
from lobes.gateway._routing import Backend, RoutingTable, infeasible_owner, list_models_payload
from lobes.roles import ROLES, build_role_registry

_PRIMARY_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_MULTIMODAL_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
_GATEWAY_URL = "http://localhost:8000"


def _full_env(**over) -> dict[str, str]:
    env = {
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _PRIMARY_ID,
        "PRIMARY_MAX_MODEL_LEN": "131072",
        "MULTIMODAL_BASE_URL": "http://vllm-multimodal:8000",
        "MULTIMODAL_SERVED_NAME": _MULTIMODAL_ID,
        "MULTIMODAL_MAX_MODEL_LEN": "32768",
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
    }
    env.update(over)
    return env


# --- FEASIBLE_ENV / build_config: the channel -------------------------------


def test_feasible_env_names_the_four_profile_scoped_backends() -> None:
    # The per-machine Profile schema covers cortex/senses/muse/embedder/
    # reranker (lobes.profiles.schema.ROLES) — mirrored 1:1 here by backend
    # name, using the SAME <PREFIX>_FEASIBLE convention as the served-context
    # overlay's <PREFIX>_MAX_MODEL_LEN (PRIMARY/MULTIMODAL/MUSE/EMBED/RERANK).
    assert FEASIBLE_ENV == {
        "primary": "PRIMARY_FEASIBLE",
        "multimodal": "MULTIMODAL_FEASIBLE",
        "muse": "MUSE_FEASIBLE",
        "embed": "EMBED_FEASIBLE",
        "rerank": "RERANK_FEASIBLE",
    }


# The opt-in muse lobe defaults to INFEASIBLE whenever it is unwired and
# unflagged (OPT_IN_BACKENDS in lobes.gateway._config) — so on every env in
# this module that doesn't wire MUSE_BASE_URL, `muse` is the honest baseline
# member of table.infeasible. Every pre-muse expectation below composes with
# this one deliberate delta.
_BASELINE = frozenset({"muse"})


def test_build_config_default_infeasible_is_empty() -> None:
    # Back-compat: no FEASIBLE var set anywhere → every existing deployment's
    # routing table is completely unaffected by this feature.
    table, _cfg = build_config(_full_env())
    assert table.infeasible == _BASELINE


def test_build_config_marks_primary_infeasible() -> None:
    table, _cfg = build_config(_full_env(PRIMARY_FEASIBLE="false"))
    assert table.infeasible == _BASELINE | {"primary"}


def test_build_config_marks_multiple_backends_infeasible() -> None:
    table, _cfg = build_config(_full_env(MULTIMODAL_FEASIBLE="false", RERANK_FEASIBLE="false"))
    assert table.infeasible == _BASELINE | {"multimodal", "rerank"}


@pytest.mark.parametrize("falsy", ["false", "False", "FALSE", "0", "no", "No", " false "])
def test_build_config_recognizes_falsy_tokens(falsy: str) -> None:
    table, _cfg = build_config(_full_env(PRIMARY_FEASIBLE=falsy))
    assert "primary" in table.infeasible


@pytest.mark.parametrize("truthy", ["true", "1", "yes", "", "TRUE"])
def test_build_config_recognizes_truthy_tokens(truthy: str) -> None:
    table, _cfg = build_config(_full_env(PRIMARY_FEASIBLE=truthy))
    assert table.infeasible == _BASELINE


def test_build_config_infeasible_independent_of_wiring() -> None:
    # A role can be declared infeasible even when its backend is never wired
    # at all (no *_BASE_URL) — the hardware fact doesn't depend on whether an
    # operator also happened to configure a container for it.
    env = {"PRIMARY_URL": "http://vllm-primary:8000", "PRIMARY_SERVED_NAME": _PRIMARY_ID}
    table, _cfg = build_config({**env, "MULTIMODAL_FEASIBLE": "false"})
    assert "multimodal" in table.infeasible
    assert not any(b.name == "multimodal" for b in table.backends)  # still unwired


# --- infeasible_owner: pure routing decision --------------------------------


def _table(infeasible: frozenset[str] = frozenset()) -> RoutingTable:
    return RoutingTable(
        backends=(
            Backend("primary", "http://vllm-primary:8000", "P"),
            Backend("multimodal", "http://vllm-multimodal:8000", "M"),
            Backend("minor", "http://vllm-minor:8000", "N"),
        ),
        default_model="P",
        aliases={"main": "P", "multimodal": "M", "minor": "N", "cortex": "P", "senses": "M"},
        infeasible=infeasible,
    )


def test_infeasible_owner_none_when_nothing_declared_infeasible() -> None:
    t = _table()
    assert infeasible_owner(t, "cortex") is None
    assert infeasible_owner(t, "P") is None
    assert infeasible_owner(t, None) is None


def test_infeasible_owner_catches_role_identity_alias() -> None:
    t = _table(frozenset({"primary"}))
    assert infeasible_owner(t, "cortex") == "primary"


@pytest.mark.parametrize("tier", ["main", "hard", "cortex"])
def test_infeasible_owner_catches_every_alias_for_the_infeasible_role(tier: str) -> None:
    t = _table(frozenset({"primary"}))
    assert infeasible_owner(t, tier) == "primary"


def test_infeasible_owner_catches_concrete_served_name() -> None:
    t = _table(frozenset({"primary"}))
    assert infeasible_owner(t, "P") == "primary"


def test_infeasible_owner_leaves_other_roles_alone() -> None:
    t = _table(frozenset({"primary"}))
    assert infeasible_owner(t, "senses") is None
    assert infeasible_owner(t, "multimodal") is None
    assert infeasible_owner(t, "minor") is None
    assert infeasible_owner(t, "M") is None


def test_infeasible_owner_unspecified_falls_through_to_default_model() -> None:
    # An unspecified request routes to default_model — if THAT backend is
    # infeasible, it must be caught too (never silently served).
    t = _table(frozenset({"primary"}))
    assert infeasible_owner(t, None) == "primary"
    assert infeasible_owner(t, "") == "primary"


def test_infeasible_owner_unknown_id_is_not_this_gates_job() -> None:
    # An id that was never advertised at all is is_unknown_model's job (404
    # model_not_found), not this gate's — infeasible_owner only fires for an
    # id/alias/tier that resolves (via the normal routing path) to a backend
    # this machine's profile named infeasible.
    t = _table(frozenset({"primary"}))
    # "never-such-id" resolves to default_model (P, infeasible) via
    # resolve_model's safety net — so this DOES fire, consistent with "an
    # unspecified/unmatched id routes to default and must not be silently
    # served if default is infeasible" above. Assert that explicitly:
    assert infeasible_owner(t, "never-such-id") == "primary"


# --- list_models_payload: infeasible is structural, not readiness-gated ----


def test_list_models_payload_excludes_infeasible_even_when_ready_true() -> None:
    # Criterion 2, at the routing layer: an infeasible-but-HEALTHY backend
    # (the live readiness cache says True) must still never be advertised.
    t = _table(frozenset({"primary"}))
    ready = {"primary": True, "multimodal": True, "minor": True}
    payload = list_models_payload(t, ready)
    ids = {entry["id"] for entry in payload["data"]}
    assert "P" not in ids
    assert {"M", "N"} <= ids


def test_list_models_payload_excludes_infeasible_with_no_readiness_signal() -> None:
    # Even with no live readiness cache wired (ready=None, the offline/unit
    # path today lists every wired backend) an infeasible backend is still
    # excluded — feasibility is a config fact, not contingent on a live probe.
    t = _table(frozenset({"multimodal"}))
    payload = list_models_payload(t)
    ids = {entry["id"] for entry in payload["data"]}
    assert "M" not in ids
    assert {"P", "N"} <= ids


def test_list_models_payload_unaffected_when_nothing_infeasible() -> None:
    t = _table()
    payload = list_models_payload(t)
    ids = {entry["id"] for entry in payload["data"]}
    assert ids == {"P", "M", "N"}


# --- handle_post: 4xx, never a silent re-route ------------------------------


class _FakeUpstream:
    def __init__(self, status: int, body: bytes = b'{"ok":1}') -> None:
        self.status = status
        self.headers = [("Content-Type", "application/json")]
        self._body = body
        self.closed = False

    def read_all(self) -> bytes:
        return self._body

    def read(self, _n: int) -> bytes:
        data, self._body = self._body, b""
        return data

    def close(self) -> None:
        self.closed = True


def _opener(behavior):
    calls: list[tuple[str, bytes]] = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append((backend.name, body))
        outcome = behavior[backend.name]
        if isinstance(outcome, Exception):
            raise outcome
        return _FakeUpstream(outcome)

    return opener, calls


def _cfg(**over):
    env = _full_env(**over)
    return build_config(env)


@pytest.mark.parametrize("requested", ["cortex", "main", "hard", _PRIMARY_ID])
def test_handle_post_rejects_infeasible_role_with_4xx(requested: str) -> None:
    table, cfg = _cfg(PRIMARY_FEASIBLE="false")
    opener, calls = _opener({"primary": 200, "multimodal": 200, "embed": 200, "rerank": 200})
    resp = S.handle_post(
        table, cfg, "/v1/chat/completions", [], json.dumps({"model": requested}).encode(), opener
    )
    assert 400 <= resp.status < 500
    assert calls == []  # never dialed ANY backend — no silent re-route
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"


def test_handle_post_infeasible_never_falls_back_to_a_different_gear() -> None:
    # The core "never served by another gear" guarantee: senses (multimodal)
    # is healthy and would happily answer, but a cortex request must not be
    # silently redirected there.
    table, cfg = _cfg(PRIMARY_FEASIBLE="false")
    opener, calls = _opener({"primary": 200, "multimodal": 200, "embed": 200, "rerank": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"cortex"}', opener)
    assert resp.status == 404
    assert calls == []
    assert "multimodal" not in [c[0] for c in calls]


def test_handle_post_feasible_roles_unaffected_by_a_sibling_infeasible_role() -> None:
    table, cfg = _cfg(PRIMARY_FEASIBLE="false")
    opener, calls = _opener({"primary": 200, "multimodal": 200, "embed": 200, "rerank": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"senses"}', opener)
    assert resp.status == 200
    assert calls[0][0] == "multimodal"


def test_handle_post_no_infeasible_roles_is_a_pure_noop() -> None:
    # Sanity: with table.infeasible empty (no FEASIBLE var set anywhere), the
    # gate never fires and existing behaviour is byte-for-byte unchanged.
    table, cfg = _cfg()
    opener, calls = _opener({"primary": 200, "multimodal": 200, "embed": 200, "rerank": 200})
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b'{"model":"cortex"}', opener)
    assert resp.status == 200
    assert calls[0][0] == "primary"


def test_handle_post_infeasible_gate_precedes_pressure_shedding() -> None:
    # Feasibility is a hardware fact, not a load condition — it takes priority
    # over (and short-circuits) the pressure-shed decision, and an
    # X-Lobes-Override cannot bypass it either (override only forces past
    # PRESSURE, never past infeasibility).
    table, cfg = _cfg(PRIMARY_FEASIBLE="false")
    opener, calls = _opener({"primary": 200, "multimodal": 200, "embed": 200, "rerank": 200})
    high_swap = {"swap_used_percent": 80.0, "iowait_percent": 0.0}
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [],
        b'{"model":"cortex"}',
        opener,
        pressure=high_swap,
        override=True,
    )
    assert resp.status == 404
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert calls == []


# --- capabilities_payload / build_role_registry: marked unserved -----------


def test_capabilities_payload_infeasible_role_is_not_advertised() -> None:
    env = _full_env(PRIMARY_FEASIBLE="false")
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert set(payload) == set(ROLES)  # still present — MARKED unserved, not omitted
    assert payload["cortex"]["feasible"] is False
    assert payload["cortex"]["ready"] is False
    # Every other role is unaffected.
    for role in ("senses", "embedder", "reranker"):
        assert payload[role]["feasible"] is True
        assert payload[role]["ready"] is True


def test_capabilities_payload_infeasible_stays_unready_even_when_backend_ready_true() -> None:
    # Criterion 2's literal proof: an infeasible-but-HEALTHY role (the live
    # readiness cache reports it True) must still never be advertised ready.
    env = _full_env(PRIMARY_FEASIBLE="false")
    table, cfg = build_config(env)
    backend_ready = {"primary": True, "multimodal": True, "embed": True, "rerank": True}
    payload = S.capabilities_payload(
        table, cfg, env=env, gateway_url=_GATEWAY_URL, backend_ready=backend_ready
    )
    assert payload["cortex"]["ready"] is False  # infeasible clamps it, despite the healthy signal
    assert payload["senses"]["ready"] is True  # unaffected sibling


def test_build_role_registry_feasible_defaults_true() -> None:
    # Back-compat: no FEASIBLE var anywhere → every gateway-fronted role
    # reports feasible=True (the field is additive, not a new failure mode).
    env = _full_env()
    table, cfg = build_config(env)
    registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    for role in ("cortex", "senses", "embedder", "reranker"):
        assert registry[role].feasible is True


def test_build_role_registry_infeasible_role_still_names_its_model() -> None:
    # "marked unserved", not hidden — a caller can still see WHAT would have
    # served this role, just that it won't (mirrors the existing "unwired"
    # convention: still name the model, loaded/ready say it isn't usable).
    env = _full_env(MULTIMODAL_FEASIBLE="false")
    table, cfg = build_config(env)
    registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert registry["senses"].model == _MULTIMODAL_ID
    assert registry["senses"].feasible is False
    assert registry["senses"].ready is False


# --- loopback: the real HTTP route, end to end ------------------------------


@pytest.fixture
def infeasible_gateway(monkeypatch):
    env = _full_env(PRIMARY_FEASIBLE="false")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    table, cfg = build_config(env)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    try:
        yield f"http://{host}:{port}"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_integration_capabilities_marks_infeasible_cortex_unserved(infeasible_gateway) -> None:
    with urllib.request.urlopen(infeasible_gateway + "/capabilities", timeout=5) as r:
        payload = json.load(r)
    assert payload["cortex"]["feasible"] is False
    assert payload["cortex"]["ready"] is False
    assert payload["senses"]["feasible"] is True
    assert payload["senses"]["ready"] is True


def test_integration_v1_models_excludes_infeasible_primary(infeasible_gateway) -> None:
    with urllib.request.urlopen(infeasible_gateway + "/v1/models", timeout=5) as r:
        payload = json.load(r)
    ids = {entry["id"] for entry in payload["data"]}
    assert _PRIMARY_ID not in ids
    assert _MULTIMODAL_ID in ids


def test_integration_post_cortex_returns_4xx_never_a_200(infeasible_gateway) -> None:
    import urllib.error

    req = urllib.request.Request(
        infeasible_gateway + "/v1/chat/completions",
        data=b'{"model":"cortex","messages":[]}',
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert 400 <= exc.value.code < 500
    body = json.loads(exc.value.read())
    assert body["error"]["type"] == "role_infeasible"


def test_integration_post_senses_still_served(infeasible_gateway) -> None:
    """A sibling role unaffected by cortex's infeasibility still gets routed —
    proves the gate is per-role, not a fleet-wide kill switch."""
    import http.client

    host_port = infeasible_gateway.removeprefix("http://")
    conn = http.client.HTTPConnection(host_port, timeout=5)
    # multimodal has no real backend listening, so this will fail to connect —
    # the point is it's NOT rejected at the gate (404 role_infeasible); it
    # reaches the (retryable) owner-down path instead.
    conn.request(
        "POST",
        "/v1/chat/completions",
        body=b'{"model":"senses","messages":[]}',
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    body = json.loads(resp.read())
    conn.close()
    assert resp.status != 404 or body.get("error", {}).get("type") != "role_infeasible"
