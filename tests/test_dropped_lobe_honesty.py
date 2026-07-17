"""Dropped-lobe honesty, end to end (brain-shapes t5, issue #113).

A mesh-brain deployment SHAPE hosts only a subset of the six Colleague-facing
roles and drops the rest to a peer box:

* ``spark-lobe`` hosts ``cortex`` + ``embedder`` + ``reranker`` + ``stt`` +
  ``tts`` and DROPS ``senses`` (the Gemma multimodal gear lives on a peer);
* ``thor-lobe`` hosts ``senses`` + ``embedder`` + ``reranker`` + ``stt`` +
  ``tts`` and DROPS ``cortex`` (the Qwen primary lives on a peer).

A dropped lobe must be HONESTLY ABSENT on the box that dropped it. This reuses
the SHIPPED #110 feasibility surface (issue #92's "advertised implies
reachable" extended to the hardware/shape dimension) with NO parallel path:

* ``lobes capabilities`` / ``GET /capabilities`` FLAG it (``feasible=false``,
  annotated) — never hidden;
* ``GET /v1/models`` OMITS it entirely;
* the generate lane returns ``404 role_infeasible`` for EVERY alias of it (its
  role-identity name, its capability-tier alias, and the back-compat synonym),
  never silently rerouting to a different, feasible gear (the #92/#91
  invariant: a request that resolves to one model is never answered by
  another).

A shape renderer (brain-shapes t3) expresses "this role is dropped" via the
SAME #110 env channel every other feasibility fact travels through — the
dropped role's ``<PREFIX>_FEASIBLE=false`` (see
:data:`lobes.gateway._config.FEASIBLE_ENV`). These tests do NOT depend on t3:
they simulate the rendered env directly, using the #110 conventions, in BOTH
shapes a real render can take —

* the dropped role's container UNWIRED (no ``*_BASE_URL`` — the realistic
  "container is simply not on this box" shape), AND
* the container wired but declared infeasible —

so the honesty invariant holds no matter how t3 chooses to render the drop.
The unwired shape is the one that used to leak: with the dropped role's
backend absent, its capability-tier aliases upward-fall-back (in
:func:`lobes.gateway._routing.tier_aliases`) to the always-present primary, so
a ``model=senses`` request would resolve to — and be silently answered by —
cortex. Closing that is exactly this task's gap (see
:func:`lobes.gateway._routing.infeasible_owner`).
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._routing import infeasible_owner, list_models_payload
from lobes.roles import ROLES, build_role_registry, role_registry_from_env

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
_GATEWAY_URL = "http://localhost:8000"

# Every alias that resolves to the DROPPED role, per box. The "careful" note in
# the task: dropped-role 404s must fire on the role-identity name, the primary
# capability tier, AND the back-compat synonym — not just the role name.
_SPARK_DROPPED_SENSES_ALIASES = ["senses", "multimodal", "normal"]
_THOR_DROPPED_CORTEX_ALIASES = ["cortex", "main", "hard", _CORTEX_ID]

# A swap/iowait sample the pressure policy treats as "shed cortex/senses to
# minor" — used to prove the pressure-degrade path can never resurrect a
# dropped role (feasibility is a hardware fact, checked before pressure).
_HIGH_PRESSURE = {"swap_used_percent": 90.0, "iowait_percent": 90.0}


# --- rendered-shape env simulators (the #110 conventions, no t3 dependency) --


def _pooling_env(**over) -> dict[str, str]:
    """embedder + reranker — hosted on BOTH mesh-lobe shapes."""
    env = {
        "EMBED_URL": "http://vllm-embed:8000",
        "EMBED_SERVED_NAME": _EMBED_ID,
        "RERANK_URL": "http://vllm-rerank:8000",
        "RERANK_SERVED_NAME": _RERANK_ID,
    }
    env.update(over)
    return env


def _spark_lobe_env(*, senses_wired: bool = False, **over) -> dict[str, str]:
    """The env a spark-lobe render produces: cortex + pooling hosted, senses DROPPED.

    The drop is expressed via ``MULTIMODAL_FEASIBLE=false`` (the #110 channel).
    ``senses_wired=False`` (default) also leaves ``MULTIMODAL_BASE_URL`` unset —
    the realistic "the Gemma container is not on this box" shape, and the one
    whose tier-alias upward-fallback used to mask the drop. ``senses_wired=True``
    pins the wired-but-infeasible variant so the invariant holds either way.
    """
    env = _pooling_env(
        PRIMARY_URL="http://vllm-primary:8000",
        PRIMARY_SERVED_NAME=_CORTEX_ID,
        PRIMARY_MAX_MODEL_LEN="131072",
        MULTIMODAL_FEASIBLE="false",
    )
    if senses_wired:
        env["MULTIMODAL_BASE_URL"] = "http://vllm-multimodal:8000"
        env["MULTIMODAL_SERVED_NAME"] = _SENSES_ID
        env["MULTIMODAL_MAX_MODEL_LEN"] = "32768"
    env.update(over)
    return env


def _thor_lobe_env(**over) -> dict[str, str]:
    """The env a thor-lobe render produces: senses + pooling hosted, cortex DROPPED.

    The drop is expressed via ``PRIMARY_FEASIBLE=false``. Note ``build_config``
    unconditionally wires the ``primary`` backend, so on thor-lobe cortex is
    wired-but-infeasible — the same shape the shipped #110 feasibility tests
    already use, which is why thor-lobe is mostly a PIN of shipped behaviour.
    """
    env = _pooling_env(
        PRIMARY_URL="http://vllm-primary:8000",
        PRIMARY_SERVED_NAME=_CORTEX_ID,
        PRIMARY_FEASIBLE="false",
        MULTIMODAL_BASE_URL="http://vllm-multimodal:8000",
        MULTIMODAL_SERVED_NAME=_SENSES_ID,
        MULTIMODAL_MAX_MODEL_LEN="32768",
    )
    env.update(over)
    return env


# --- a fake upstream so a POST that DOES route can be observed ---------------


class _FakeUpstream:
    def __init__(self, status: int, body: bytes = b'{"ok":1}') -> None:
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


def _opener(behavior: dict[str, int]):
    """An ``open_upstream`` stub recording every backend it is asked to dial.

    Every wired generate/pooling backend answers 200 by default, so a dropped
    role that leaked would visibly succeed (200) — the test then proves it was
    rejected at the gate (404) with ``calls == []`` instead.
    """
    calls: list[str] = []

    def opener(backend, path, body, headers, *, connect_timeout, read_timeout):
        calls.append(backend.name)
        return _FakeUpstream(behavior.get(backend.name, 200))

    return opener, calls


def _post(table, cfg, model: str, *, pressure=None, override=False):
    opener, calls = _opener({"primary": 200, "multimodal": 200, "embed": 200, "rerank": 200})
    resp = S.handle_post(
        table,
        cfg,
        "/v1/chat/completions",
        [],
        json.dumps({"model": model}).encode(),
        opener,
        pressure=pressure,
        override=override,
    )
    return resp, calls


# ============================================================================
# Criterion 1 — spark-lobe drops senses: flagged, omitted, 404 on every alias
# ============================================================================


@pytest.mark.parametrize("senses_wired", [False, True])
def test_spark_capabilities_flags_dropped_senses_not_hidden(senses_wired: bool) -> None:
    env = _spark_lobe_env(senses_wired=senses_wired)
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    # Present, not omitted — MARKED unserved (the #92/#110 "flag, don't hide").
    assert set(payload) == set(ROLES)
    assert payload["senses"]["feasible"] is False
    assert payload["senses"]["ready"] is False
    assert payload["senses"]["model"]  # still NAMES what would have served it
    # Every hosted role is unaffected.
    for role in ("cortex", "embedder", "reranker"):
        assert payload[role]["feasible"] is True


@pytest.mark.parametrize("senses_wired", [False, True])
def test_spark_v1_models_omits_dropped_senses(senses_wired: bool) -> None:
    table, _cfg = build_config(_spark_lobe_env(senses_wired=senses_wired))
    # Even with a live readiness cache calling every wired backend healthy, the
    # dropped role is never advertised.
    ready = {"primary": True, "multimodal": True, "embed": True, "rerank": True}
    ids = {e["id"] for e in list_models_payload(table, ready)["data"]}
    assert _SENSES_ID not in ids
    assert _CORTEX_ID in ids
    assert {_EMBED_ID, _RERANK_ID} <= ids


@pytest.mark.parametrize("senses_wired", [False, True])
@pytest.mark.parametrize("alias", _SPARK_DROPPED_SENSES_ALIASES)
def test_spark_post_dropped_senses_alias_returns_404_role_infeasible(
    alias: str, senses_wired: bool
) -> None:
    table, cfg = build_config(_spark_lobe_env(senses_wired=senses_wired))
    resp, calls = _post(table, cfg, alias)
    assert resp.status == 404
    assert calls == []  # never dialed ANY backend — no silent re-route
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    # The 404 names the requested role and says it is not served on this box.
    assert alias in body["error"]["message"]
    assert "never be served here" in body["error"]["message"]


def test_spark_infeasible_owner_catches_dropped_senses_aliases_even_unwired() -> None:
    # The routing-layer truth behind the POST 404s: with senses' container
    # absent, its aliases must STILL resolve to the infeasible multimodal
    # backend name, never upward-fall-back to the (feasible) primary.
    table, _cfg = build_config(_spark_lobe_env(senses_wired=False))
    assert not any(b.name == "multimodal" for b in table.backends)  # container absent
    for alias in _SPARK_DROPPED_SENSES_ALIASES:
        assert infeasible_owner(table, alias) == "multimodal", alias
    # A hosted role's alias is never caught.
    assert infeasible_owner(table, "cortex") is None
    assert infeasible_owner(table, "main") is None


# ============================================================================
# Criterion 2 — thor-lobe drops cortex: mirror assertions
# ============================================================================


def test_thor_capabilities_flags_dropped_cortex_not_hidden() -> None:
    env = _thor_lobe_env()
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert set(payload) == set(ROLES)
    assert payload["cortex"]["feasible"] is False
    assert payload["cortex"]["ready"] is False
    assert payload["cortex"]["model"]  # still named
    for role in ("senses", "embedder", "reranker"):
        assert payload[role]["feasible"] is True


def test_thor_v1_models_omits_dropped_cortex() -> None:
    table, _cfg = build_config(_thor_lobe_env())
    ready = {"primary": True, "multimodal": True, "embed": True, "rerank": True}
    ids = {e["id"] for e in list_models_payload(table, ready)["data"]}
    assert _CORTEX_ID not in ids
    assert _SENSES_ID in ids
    assert {_EMBED_ID, _RERANK_ID} <= ids


@pytest.mark.parametrize("alias", _THOR_DROPPED_CORTEX_ALIASES)
def test_thor_post_dropped_cortex_alias_returns_404_role_infeasible(alias: str) -> None:
    table, cfg = build_config(_thor_lobe_env())
    resp, calls = _post(table, cfg, alias)
    assert resp.status == 404
    assert calls == []
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert alias in body["error"]["message"]


def test_thor_infeasible_owner_catches_dropped_cortex_aliases() -> None:
    table, _cfg = build_config(_thor_lobe_env())
    for alias in _THOR_DROPPED_CORTEX_ALIASES:
        assert infeasible_owner(table, alias) == "primary", alias
    # senses is hosted on thor — never caught.
    assert infeasible_owner(table, "senses") is None
    assert infeasible_owner(table, "multimodal") is None


# ============================================================================
# Criterion 3 — a dropped role is NEVER silently rerouted to a different model
# ============================================================================


def test_dropped_senses_never_answered_by_healthy_cortex() -> None:
    # The headline guarantee: cortex is healthy and would happily answer, but a
    # senses request on spark-lobe must NOT be silently redirected to it.
    table, cfg = build_config(_spark_lobe_env(senses_wired=False))
    resp, calls = _post(table, cfg, "senses")
    assert resp.status == 404
    assert "primary" not in calls  # cortex never touched
    assert calls == []  # in fact nothing was touched — the 404 is terminal


def test_dropped_cortex_never_answered_by_healthy_senses() -> None:
    # Mirror: senses is healthy on thor-lobe, but a cortex request must not be
    # silently answered by it.
    table, cfg = build_config(_thor_lobe_env())
    resp, calls = _post(table, cfg, "cortex")
    assert resp.status == 404
    assert "multimodal" not in calls
    assert calls == []


@pytest.mark.parametrize("senses_wired", [False, True])
@pytest.mark.parametrize("alias", _SPARK_DROPPED_SENSES_ALIASES)
def test_pressure_degrade_cannot_resurrect_dropped_senses(alias: str, senses_wired: bool) -> None:
    # Under swap/iowait pressure the policy degrades cortex/senses requests to
    # `minor`. That degrade must NEVER resurrect a DROPPED role: feasibility is
    # a hardware/shape fact, checked BEFORE pressure, and not even
    # X-Lobes-Override (which forces past pressure) can bypass it.
    table, cfg = build_config(_spark_lobe_env(senses_wired=senses_wired))
    resp, calls = _post(table, cfg, alias, pressure=_HIGH_PRESSURE, override=True)
    assert resp.status == 404
    assert calls == []
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"


@pytest.mark.parametrize("alias", ["cortex", "main", "hard"])
def test_pressure_degrade_cannot_resurrect_dropped_cortex(alias: str) -> None:
    table, cfg = build_config(_thor_lobe_env())
    resp, calls = _post(table, cfg, alias, pressure=_HIGH_PRESSURE, override=True)
    assert resp.status == 404
    assert calls == []
    assert json.loads(resp.body)["error"]["type"] == "role_infeasible"


def test_hosted_sibling_still_routes_on_spark_lobe() -> None:
    # The gate is per-role, not a fleet-wide kill switch: cortex (hosted on
    # spark-lobe) still routes to its own backend, and the pooling gears too.
    table, cfg = build_config(_spark_lobe_env(senses_wired=False))
    for alias, backend in (("cortex", "primary"), ("main", "primary")):
        resp, calls = _post(table, cfg, alias)
        assert resp.status == 200, alias
        assert calls == [backend], alias


def test_dropped_shapes_never_mark_audio_infeasible() -> None:
    # stt/tts are hosted on BOTH mesh-lobe shapes; the per-machine Profile
    # schema doesn't cover them, so no FEASIBLE channel can ever flip them.
    for env in (_spark_lobe_env(AUDIO_URL="http://realtime:8080"), _thor_lobe_env()):
        table, cfg = build_config(env)
        payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
        assert payload["stt"]["feasible"] is True
        assert payload["tts"]["feasible"] is True


# ============================================================================
# Criterion 4 — both surfaces derive the SAME feasible:false from the SAME env
# ============================================================================


def _feasible_by_role(registry) -> dict[str, bool]:
    return {role: registry[role].feasible for role in ROLES}


def test_spark_cli_and_gateway_agree_on_dropped_senses() -> None:
    # The #110 lesson: one builder, don't let two callers feed it different
    # config. The gateway's capabilities and the CLI's offline registry are
    # both `build_role_registry` over the SAME env — so their feasibility truth
    # must agree, role for role, with senses the single dropped one.
    env = _spark_lobe_env(senses_wired=False)
    table, cfg = build_config(env)
    gateway = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    cli = role_registry_from_env(env, gateway_url=_GATEWAY_URL)  # the CLI's offline path
    assert _feasible_by_role(gateway) == _feasible_by_role(cli)
    assert _feasible_by_role(cli) == {
        "cortex": True,
        "senses": False,
        "muse": False,
        "embedder": True,
        "reranker": True,
        "stt": True,
        "tts": True,
    }


def test_thor_cli_and_gateway_agree_on_dropped_cortex() -> None:
    env = _thor_lobe_env()
    table, cfg = build_config(env)
    gateway = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    cli = role_registry_from_env(env, gateway_url=_GATEWAY_URL)
    assert _feasible_by_role(gateway) == _feasible_by_role(cli)
    assert _feasible_by_role(cli) == {
        "cortex": False,
        "senses": True,
        "muse": False,
        "embedder": True,
        "reranker": True,
        "stt": True,
        "tts": True,
    }


# ============================================================================
# Loopback — the real HTTP route, end to end, for a rendered spark-lobe env
# ============================================================================


@pytest.fixture
def spark_lobe_gateway(monkeypatch):
    """A real gateway serving a spark-lobe env (senses dropped, container absent)."""
    env = _spark_lobe_env(senses_wired=False)
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


def test_integration_spark_capabilities_flags_dropped_senses(spark_lobe_gateway) -> None:
    with urllib.request.urlopen(spark_lobe_gateway + "/capabilities", timeout=5) as r:
        payload = json.load(r)
    assert payload["senses"]["feasible"] is False
    assert payload["senses"]["ready"] is False
    assert payload["cortex"]["feasible"] is True


def test_integration_spark_v1_models_omits_dropped_senses(spark_lobe_gateway) -> None:
    with urllib.request.urlopen(spark_lobe_gateway + "/v1/models", timeout=5) as r:
        payload = json.load(r)
    ids = {e["id"] for e in payload["data"]}
    assert _SENSES_ID not in ids
    assert _CORTEX_ID in ids


@pytest.mark.parametrize("alias", _SPARK_DROPPED_SENSES_ALIASES)
def test_integration_spark_post_dropped_senses_alias_is_404(spark_lobe_gateway, alias: str) -> None:
    req = urllib.request.Request(
        spark_lobe_gateway + "/v1/chat/completions",
        data=json.dumps({"model": alias, "messages": []}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404
    body = json.loads(exc.value.read())
    assert body["error"]["type"] == "role_infeasible"
