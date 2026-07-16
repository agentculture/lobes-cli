"""The contract-test MATRIX for the reference deployment shapes (mesh-brain t4, issue #112).

One systematic, DATA-DRIVEN enumeration of every (built-in shape, dropped core
role) cell — derived from the shipped shape TOMLs' ``hosts`` lists themselves
(:func:`lobes.profiles.shapes.builtin_shape_names`), so a future built-in shape
is covered automatically the moment its TOML lands. Today the cells are:

* ``spark-lobe``  drops ``senses``            (the Gemma gear lives on a peer);
* ``thor-lobe``   drops ``cortex``            (the Qwen primary lives on a peer);
* ``thor-muse``   drops BOTH heavies (``cortex`` AND ``senses``) — its generate
  lane is the opt-in 31B ``muse`` lobe;
* ``orin-small``  drops BOTH heavies (``cortex`` AND ``senses``) — its generate
  lane is the opt-in 4B ``minor`` gear;
* ``machine-as-brain`` drops no DEFAULT role; since the opt-in core ``muse``
  lobe landed it contributes exactly one cell (``muse`` — like every shape
  that doesn't host it), while its default path stays regression-pinned by the
  golden suites (see below).

For EVERY cell, the honesty contract shipped by #110/#113-t5 and extended by
mesh-brain t3 is asserted in full:

1. **capabilities** — the gateway payload (``GET /capabilities``) and the CLI's
   offline registry (``lobes capabilities``) BOTH flag the dropped role
   ``feasible: false`` (present and marked, never hidden);
2. **GET /v1/models** — the dropped role's model id is omitted, even with every
   wired backend reporting ready;
3. **requests** — EVERY alias that maps to the dropped role (its role-identity
   name, its capability tier, and the back-compat synonym — derived from
   :data:`lobes.catalog.TIER_ROLE`, never hand-listed) 404s ``role_infeasible``
   without dialing any backend; and with peer origins declared
   (``<PREFIX>_PEER_ORIGIN``, mesh-brain t3) the 404 body carries the referral
   (``hosted_by``) naming the RIGHT peer per role.

The per-shape suites that predate this matrix (``tests/test_dropped_lobe_honesty.py``
for spark-lobe/thor-lobe, ``tests/test_peer_referral.py`` for spark-lobe's
referrals) stay as-is; this module's job is that no cell — orin-small's two
drops especially — is covered only by analogy.

**The machine-as-brain default-path regression** rides on the golden suites
(all run unconditionally in CI's ``test`` job — ``.github/workflows/tests.yml``
runs the full pytest suite):

* ``tests/test_shape_goldens.py::test_machine_as_brain_is_byte_identical_to_profile_golden``
  byte-diffs the identity shape against the pre-shape per-card goldens;
* ``tests/test_profile_goldens.py`` byte-diffs ``tests/goldens/<card>.env`` +
  ``template-defaults.env``;
* ``tests/test_shape_goldens.py::test_shape_golden_byte_for_byte`` byte-diffs
  every (shape, card) golden under ``tests/goldens/shapes/``;
* ``tests/test_init_shape.py::test_bare_init_env_equals_explicit_machine_as_brain_shape``
  pins the bare-init path to the identity shape.

This module adds the one golden-adjacent guard those suites cannot express:
the t1 full-native reclaim VALUES pinned as explicit assertions against the
golden FILES (not just golden-file equality), so a future golden regeneration
cannot silently lower them.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from lobes.catalog import TIER_ROLE
from lobes.gateway import server as S
from lobes.gateway._config import FEASIBLE_ENV, PEER_ORIGIN_ENV, build_config
from lobes.gateway._routing import list_models_payload
from lobes.profiles.loader import resolve_profile
from lobes.profiles.schema import ROLES as CORE_ROLES
from lobes.profiles.shapes import (
    OPT_IN_CORE_ROLES,
    Shape,
    builtin_shape_names,
    resolve_shape,
)
from lobes.roles import ROLE_BACKEND, ROLES, build_role_registry, role_registry_from_env

_GATEWAY_URL = "http://localhost:8000"

# One distinct model id per core role, plus the opt-in minor gear — distinct so
# an omission/leak on /v1/models is attributable to exactly one role.
_MODEL_ID: dict[str, str] = {
    "cortex": "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP",
    "senses": "coolthor/gemma-4-12B-it-NVFP4A16",
    "muse": "nvidia/Gemma-4-31B-IT-NVFP4",
    "embedder": "Qwen/Qwen3-Embedding-0.6B",
    "reranker": "Qwen/Qwen3-Reranker-0.6B",
}
_MINOR_ID = "Qwen/Qwen3.5-4B"

# role -> the (url_key, name_key) env pair that WIRES its gateway backend —
# mirrors the gateway service's environment block in the fleet compose template
# (and _config.build_config's _optional_backend keys).
_WIRE_ENV: dict[str, tuple[str, str]] = {
    "cortex": ("PRIMARY_URL", "PRIMARY_SERVED_NAME"),
    "senses": ("MULTIMODAL_BASE_URL", "MULTIMODAL_SERVED_NAME"),
    "muse": ("MUSE_BASE_URL", "MUSE_SERVED_NAME"),
    "embedder": ("EMBED_URL", "EMBED_SERVED_NAME"),
    "reranker": ("RERANK_URL", "RERANK_SERVED_NAME"),
}


def _peer_origin(role: str) -> str:
    """A distinct, operator-declared peer origin per role (never derived, #92)."""
    return f"http://peer-of-{role}.local:8001"


def _dropped_core_roles(shape: Shape) -> tuple[str, ...]:
    """The core roles this shape does NOT host — the matrix's cell source."""
    return tuple(role for role in CORE_ROLES if not shape.hosts_role(role))


def _aliases(role: str) -> tuple[str, ...]:
    """EVERY alias that maps to ``role``'s backend, derived from TIER_ROLE.

    For cortex: ``cortex``/``main``/``hard``; for senses:
    ``senses``/``multimodal``/``normal``. Derived (not hand-listed) so a new
    alias in the catalog is covered here automatically.
    """
    backend = ROLE_BACKEND[role]
    return tuple(sorted(alias for alias, be in TIER_ROLE.items() if be == backend))


# The matrix itself: every (built-in shape, dropped core role) cell, derived
# from the shipped shape data. A future built-in shape's drops land here with
# zero edits to this module.
CELLS: tuple[tuple[str, str], ...] = tuple(
    (shape_name, role)
    for shape_name in builtin_shape_names()
    for role in _dropped_core_roles(resolve_shape(shape_name))
)

# Cells whose dropped role has generate-lane aliases (all of today's cells —
# only cortex/senses are droppable heavies with TIER_ROLE aliases).
ALIAS_CELLS: tuple[tuple[str, str, str], ...] = tuple(
    (shape_name, role, alias) for shape_name, role in CELLS for alias in _aliases(role)
)

_CELL_IDS = [f"{shape}--{role}" for shape, role in CELLS]
_ALIAS_CELL_IDS = [f"{shape}--{role}--{alias}" for shape, role, alias in ALIAS_CELLS]


def test_matrix_enumerates_the_documented_reference_cells() -> None:
    """The derived matrix IS the documented one — the explicit enumeration.

    spark-lobe drops senses; thor-lobe drops cortex; thor-muse and orin-small
    drop BOTH heavies; every shape that doesn't host the opt-in muse lobe
    (all but thor-muse) contributes a muse cell too. If this fails because a NEW
    built-in shape landed: the parametrized cell tests below already cover it —
    just extend this documented set to match its declared drops.
    """
    assert set(CELLS) == {
        ("machine-as-brain", "muse"),
        ("orin-small", "cortex"),
        ("orin-small", "senses"),
        ("orin-small", "muse"),
        ("spark-lobe", "senses"),
        ("spark-lobe", "muse"),
        ("thor-lobe", "cortex"),
        ("thor-lobe", "muse"),
        ("thor-muse", "cortex"),
        ("thor-muse", "senses"),
    }
    # machine-as-brain contributes ONLY the opt-in muse cell — never a
    # default-role drop (its default path is golden-pinned instead).
    assert all(role == "muse" for shape, role in CELLS if shape == "machine-as-brain")


def _gateway_env(shape: Shape, *, peers: bool = False) -> dict[str, str]:
    """The gateway env a rendered ``shape`` produces, per the #110 conventions.

    A hosted core role's backend is wired (its ``*_URL``/``*_SERVED_NAME``
    pair); a dropped role renders ONLY its ``<PREFIX>_FEASIBLE=false`` marker
    and stays unwired — the realistic "the container is simply not on this box"
    shape (the primary is the exception: ``build_config`` wires it
    unconditionally, so a dropped cortex is wired-but-infeasible, exactly as on
    a real thor-lobe/orin-small box). ``peers=True`` adds the opt-in mesh-brain
    t3 referral config: one distinct ``<PREFIX>_PEER_ORIGIN`` per dropped role.
    """
    env: dict[str, str] = {
        # build_config wires the primary unconditionally; give it its real id
        # so the wired-but-infeasible dropped-cortex case is faithfully shaped.
        "PRIMARY_URL": "http://vllm-primary:8000",
        "PRIMARY_SERVED_NAME": _MODEL_ID["cortex"],
    }
    for role in CORE_ROLES:
        backend = ROLE_BACKEND[role]
        if shape.hosts_role(role):
            url_key, name_key = _WIRE_ENV[role]
            env[url_key] = f"http://vllm-{backend}:8000"
            env[name_key] = _MODEL_ID[role]
        else:
            # A non-hosted OPT-IN core role (muse) renders NO marker on a
            # silent card (see shape_render.compose_profile) — the gateway's
            # OPT_IN_BACKENDS unwired-by-default rule makes it infeasible
            # anyway, and THAT is the realistic stale/silent-card shape this
            # matrix must prove honest.
            if role not in OPT_IN_CORE_ROLES:
                env[FEASIBLE_ENV[backend]] = "false"
            if peers:
                env[PEER_ORIGIN_ENV[backend]] = _peer_origin(role)
    if shape.hosts_role("minor"):
        env["MINOR_BASE_URL"] = "http://vllm-minor:8000"
        env["MINOR_SERVED_NAME"] = _MINOR_ID
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


def _post(table, cfg, model: str, path: str = "/v1/chat/completions"):
    opener, calls = _opener()
    resp = S.handle_post(table, cfg, path, [], json.dumps({"model": model}).encode(), opener)
    return resp, calls


def _all_ready(table) -> dict[str, bool]:
    """A readiness snapshot calling every WIRED backend healthy — the worst case
    for honesty: even then a dropped role must never be advertised."""
    return {backend.name: True for backend in table.backends}


# ============================================================================
# Cell assertion 1 — capabilities flag the dropped role, on BOTH surfaces
# ============================================================================


@pytest.mark.parametrize("shape_name,role", CELLS, ids=_CELL_IDS)
def test_cell_capabilities_flag_dropped_role_on_gateway_and_cli(shape_name: str, role: str) -> None:
    shape = resolve_shape(shape_name)
    env = _gateway_env(shape)
    table, cfg = build_config(env)
    # The gateway's GET /capabilities payload: present, marked — never hidden.
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert set(payload) == set(ROLES)
    assert payload[role]["feasible"] is False
    assert payload[role]["ready"] is False
    assert payload[role]["model"]  # still NAMES what would have served it
    # Every hosted core role is unaffected (audio roles are always feasible).
    for hosted in CORE_ROLES:
        if shape.hosts_role(hosted):
            assert payload[hosted]["feasible"] is True, hosted
    # The CLI's offline registry (`lobes capabilities`) agrees, role for role.
    gateway_registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    cli_registry = role_registry_from_env(env, gateway_url=_GATEWAY_URL)
    for r in ROLES:
        assert cli_registry[r].feasible == gateway_registry[r].feasible, r
    assert cli_registry[role].feasible is False


# ============================================================================
# Cell assertion 2 — GET /v1/models omits the dropped role
# ============================================================================


@pytest.mark.parametrize("shape_name,role", CELLS, ids=_CELL_IDS)
def test_cell_v1_models_omits_dropped_role(shape_name: str, role: str) -> None:
    shape = resolve_shape(shape_name)
    table, _cfg = build_config(_gateway_env(shape))
    ids = {e["id"] for e in list_models_payload(table, _all_ready(table))["data"]}
    assert _MODEL_ID[role] not in ids
    # Every hosted core role's id IS advertised.
    for hosted in CORE_ROLES:
        if shape.hosts_role(hosted):
            assert _MODEL_ID[hosted] in ids, hosted


# ============================================================================
# Cell assertion 3 — every alias 404s role_infeasible; referral when declared
# ============================================================================


@pytest.mark.parametrize("shape_name,role,alias", ALIAS_CELLS, ids=_ALIAS_CELL_IDS)
def test_cell_every_alias_404s_role_infeasible_dialing_nothing(
    shape_name: str, role: str, alias: str
) -> None:
    table, cfg = build_config(_gateway_env(resolve_shape(shape_name)))
    resp, calls = _post(table, cfg, alias)
    assert resp.status == 404
    assert calls == []  # never dialed ANY backend — no silent re-route
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["code"] == "role_infeasible"
    assert alias in body["error"]["message"]
    assert "hosted_by" not in body["error"]  # no peer declared -> no referral


@pytest.mark.parametrize("shape_name,role,alias", ALIAS_CELLS, ids=_ALIAS_CELL_IDS)
def test_cell_404_carries_the_right_peer_referral_when_declared(
    shape_name: str, role: str, alias: str
) -> None:
    # Each dropped role gets a DISTINCT declared peer — on orin-small (two
    # drops) this also proves the referral is per-backend, never mixed up.
    table, cfg = build_config(_gateway_env(resolve_shape(shape_name), peers=True))
    resp, calls = _post(table, cfg, alias)
    assert resp.status == 404
    assert calls == []  # the referral is an ANSWER, never a forward (no proxy)
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _peer_origin(role)
    assert _peer_origin(role) in body["error"]["message"]


@pytest.mark.parametrize("shape_name,role", CELLS, ids=_CELL_IDS)
def test_cell_capabilities_carry_the_right_peer_referral_when_declared(
    shape_name: str, role: str
) -> None:
    shape = resolve_shape(shape_name)
    env = _gateway_env(shape, peers=True)
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert payload[role]["hosted_by"] == _peer_origin(role)
    assert payload[role]["feasible"] is False
    for hosted in ROLES:
        if hosted in CORE_ROLES and not shape.hosts_role(hosted):
            continue
        assert "hosted_by" not in payload[hosted], hosted


@pytest.mark.parametrize("shape_name,role", CELLS, ids=_CELL_IDS)
def test_cell_concrete_model_id_is_never_served_4xx_without_a_dial(
    shape_name: str, role: str
) -> None:
    """The dropped role's CONCRETE served id also 4xxes, wired or not.

    A dropped role whose backend is still wired (cortex — the primary is wired
    unconditionally) 404s ``role_infeasible``; one whose backend is absent
    (senses) was never advertised, so its id 404s ``model_not_found``. Either
    way: an honest 4xx, no backend dialed, never a 200 from a different gear.
    """
    table, cfg = build_config(_gateway_env(resolve_shape(shape_name)))
    resp, calls = _post(table, cfg, _MODEL_ID[role])
    assert resp.status == 404
    assert calls == []
    wired = any(b.name == ROLE_BACKEND[role] for b in table.backends)
    expected_type = "role_infeasible" if wired else "model_not_found"
    assert json.loads(resp.body)["error"]["type"] == expected_type


# ============================================================================
# The hosted lanes stay live on every shape (the gate is per-role, not global)
# ============================================================================


@pytest.mark.parametrize("shape_name", sorted({shape for shape, _ in CELLS}))
def test_hosted_generate_lane_still_routes_on_every_mesh_shape(shape_name: str) -> None:
    """Whatever generate gear a shape hosts still answers: cortex on spark-lobe,
    senses on thor-lobe, and the opt-in 4B minor on orin-small (both tier
    vocabularies)."""
    shape = resolve_shape(shape_name)
    table, cfg = build_config(_gateway_env(shape))
    expected: list[tuple[str, str]] = []
    if shape.hosts_role("cortex"):
        expected += [("cortex", "primary"), ("main", "primary"), ("hard", "primary")]
    if shape.hosts_role("senses"):
        expected += [
            ("senses", "multimodal"),
            ("multimodal", "multimodal"),
            ("normal", "multimodal"),
        ]
    if shape.hosts_role("muse"):
        expected += [("muse", "muse")]
    if shape.hosts_role("minor"):
        expected += [("minor", "minor"), ("cheap", "minor")]
    assert expected, f"{shape_name} hosts no generate lane at all?"
    for alias, backend in expected:
        resp, calls = _post(table, cfg, alias)
        assert resp.status == 200, (shape_name, alias)
        assert calls == [backend], (shape_name, alias)


@pytest.mark.parametrize("shape_name", sorted({shape for shape, _ in CELLS}))
def test_hosted_pooling_lanes_still_route_on_every_mesh_shape(shape_name: str) -> None:
    shape = resolve_shape(shape_name)
    table, cfg = build_config(_gateway_env(shape))
    if shape.hosts_role("embedder"):
        resp, calls = _post(table, cfg, _MODEL_ID["embedder"], path="/v1/embeddings")
        assert (resp.status, calls) == (200, ["embed"]), shape_name
    if shape.hosts_role("reranker"):
        resp, calls = _post(table, cfg, _MODEL_ID["reranker"], path="/v1/rerank")
        assert (resp.status, calls) == (200, ["rerank"]), shape_name


def test_orin_small_unspecified_model_404s_the_infeasible_default_honestly() -> None:
    """On orin-small the gateway's default model is still the (dropped) cortex id,
    so a request with NO model field 404s ``role_infeasible`` — honest, and by
    design: silently answering the default identity with the 4B minor would be
    exactly the #91/#92 "answered by a model you did not ask for" violation.
    With peers declared the 404 carries cortex's referral, so the caller knows
    where the brain's default lane actually lives."""
    table, cfg = build_config(_gateway_env(resolve_shape("orin-small"), peers=True))
    opener, calls = _opener()
    resp = S.handle_post(table, cfg, "/v1/chat/completions", [], b"{}", opener)
    assert resp.status == 404
    assert calls == []
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _peer_origin("cortex")


# ============================================================================
# Loopback — the real HTTP gateway, one server per cell, no outbound ever
# ============================================================================


@pytest.mark.parametrize("shape_name,role", CELLS, ids=_CELL_IDS)
def test_cell_end_to_end_over_real_http(shape_name: str, role: str, monkeypatch) -> None:
    """GET /capabilities flags + refers, GET /v1/models omits, POST 404s with the
    referral — through a real ThreadingHTTPServer, with ``open_upstream`` (the
    gateway's ONLY outbound seam) replaced by a tripwire."""
    env = _gateway_env(resolve_shape(shape_name), peers=True)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    outbound: list[str] = []

    def no_outbound(backend, *a, **k):
        outbound.append(backend.base_url)
        raise AssertionError(f"gateway opened an outbound connection to {backend.base_url}")

    monkeypatch.setattr(S, "open_upstream", no_outbound)
    table, cfg = build_config(env)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address
    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(base + "/capabilities", timeout=5) as r:
            payload = json.load(r)
        assert payload[role]["feasible"] is False
        assert payload[role]["hosted_by"] == _peer_origin(role)

        with urllib.request.urlopen(base + "/v1/models", timeout=5) as r:
            ids = {e["id"] for e in json.load(r)["data"]}
        assert _MODEL_ID[role] not in ids

        req = urllib.request.Request(
            base + "/v1/chat/completions",
            data=json.dumps({"model": role, "messages": []}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(req, timeout=5)
        assert exc.value.code == 404
        body = json.loads(exc.value.read())
        assert body["error"]["type"] == "role_infeasible"
        assert body["error"]["hosted_by"] == _peer_origin(role)
        assert outbound == []  # answered locally, never forwarded (issue #115)
    finally:
        httpd.shutdown()
        httpd.server_close()


# ============================================================================
# t1 pinned regressions — the full-native reclaim values, asserted against the
# golden FILES so a future golden regeneration cannot silently lower them
# ============================================================================

_SHAPES_GOLDENS_DIR = Path(__file__).resolve().parent / "goldens" / "shapes"


def _golden_env(name: str) -> dict[str, str]:
    text = (_SHAPES_GOLDENS_DIR / f"{name}.env").read_text(encoding="utf-8")
    return dict(line.split("=", 1) for line in text.splitlines() if line)


def test_spark_lobe_golden_pins_cortex_full_native_reclaim() -> None:
    """spark-lobe on the GB10: cortex serves its full native 256K, with a GPU
    budget STRICTLY above the co-resident 0.30 (measured reclaim 0.44 — the
    2026-07-14 live validation; issue #113). Asserted against the golden file's
    own bytes, on top of the golden byte-diff, so `regen.py` cannot silently
    regress the shipped values."""
    env = _golden_env("spark-lobe__spark")
    assert env["PRIMARY_MAX_MODEL_LEN"] == "262144"
    util = float(env["PRIMARY_GPU_MEM_UTIL"])
    assert util > 0.30  # the non-negotiable floor: strictly above co-resident
    # ... and strictly above whatever the card profile currently ships, so the
    # pin survives a co-resident retune too.
    assert util > resolve_profile("spark").role("cortex").gpu_mem_util


def test_thor_lobe_golden_pins_senses_full_native_reclaim() -> None:
    """thor-lobe on the AGX Thor: senses serves its full native 128K, with a GPU
    budget STRICTLY above the co-resident 0.14 (measured reclaim 0.30 — the
    2026-07-14 live validation; issue #113)."""
    env = _golden_env("thor-lobe__thor")
    assert env["MULTIMODAL_MAX_MODEL_LEN"] == "131072"
    util = float(env["MULTIMODAL_GPU_MEM_UTIL"])
    assert util > 0.14
    assert util > resolve_profile("thor").role("senses").gpu_mem_util
