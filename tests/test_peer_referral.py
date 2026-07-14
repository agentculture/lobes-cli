"""Honest referral: opt-in peer config on the gateway (mesh-brain t3, issue #112).

A mesh-brain box drops a role to a peer box. With **peer config** set — the
operator-declared ``<PREFIX>_PEER_ORIGIN`` env vars (see
:data:`lobes.gateway._config.PEER_ORIGIN_ENV`) — the box's honesty surfaces
name the peer that actually hosts each unhosted role:

* ``GET /capabilities`` / ``lobes capabilities`` annotate the unhosted role
  with ``hosted_by: <peer origin>``;
* the ``404 role_infeasible`` body carries the referral (``hosted_by`` in the
  error object, and the origin in the message).

Two invariants bound the feature:

* **Byte-identity with no peer config** — an operator who sets nothing gets
  responses byte-identical to the pre-change contract (regression-pinned
  below against the exact pre-change bytes).
* **NO data-plane proxying** — the gateway never forwards a request to a
  peer. A request for an unhosted role is answered locally with the 404
  referral and zero outbound connections (proven below at both the
  ``handle_post`` seam and the real HTTP loopback). Proxy-lobes (following
  the referral) is explicitly deferred — issue #115.

The referral origin is OPERATOR-DECLARED, never derived from hostnames or
interfaces (the #92 lesson: never fabricate an absolute URL).
"""

from __future__ import annotations

import dataclasses
import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lobes.cli import main
from lobes.gateway import server as S
from lobes.gateway._config import FEASIBLE_ENV, PEER_ORIGIN_ENV, build_config
from lobes.gateway._routing import list_models_payload
from lobes.roles import ROLES, annotate_peer_referrals, build_role_registry
from lobes.runtime import _compose, _env

_CORTEX_ID = "sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP"
_SENSES_ID = "coolthor/gemma-4-12B-it-NVFP4A16"
_EMBED_ID = "Qwen/Qwen3-Embedding-0.6B"
_RERANK_ID = "Qwen/Qwen3-Reranker-0.6B"
_GATEWAY_URL = "http://localhost:8000"

# The peer origins an operator would declare — full, dialable origins. These
# are DECLARED per box in .env, never derived (#92).
_THOR_ORIGIN = "http://thor.local:8001"
_SPARK_ORIGIN = "http://spark.local:8001"


def _spark_lobe_env(*, peers: bool = False, **over) -> dict[str, str]:
    """A rendered spark-lobe env: cortex + pooling hosted, senses DROPPED.

    ``peers=True`` adds the opt-in referral config: the operator declares the
    peer box (Thor) that hosts the dropped ``senses`` role.
    """
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
    if peers:
        env["MULTIMODAL_PEER_ORIGIN"] = _THOR_ORIGIN
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


def _post(table, cfg, model: str, path: str = "/v1/chat/completions"):
    opener, calls = _opener()
    resp = S.handle_post(table, cfg, path, [], json.dumps({"model": model}).encode(), opener)
    return resp, calls


# ============================================================================
# Peer-config parsing (the opt-in surface)
# ============================================================================


def test_peer_origin_env_mirrors_feasible_env_prefixes() -> None:
    # One "<PREFIX>_<KNOB>" convention to learn: the peer-origin channel names
    # exactly the backends the feasibility channel names.
    assert set(PEER_ORIGIN_ENV) == set(FEASIBLE_ENV)
    assert PEER_ORIGIN_ENV["multimodal"] == "MULTIMODAL_PEER_ORIGIN"
    assert PEER_ORIGIN_ENV["primary"] == "PRIMARY_PEER_ORIGIN"


def test_build_config_default_peer_origins_is_empty() -> None:
    table, _cfg = build_config(_spark_lobe_env())
    assert dict(table.peer_origins) == {}


def test_build_config_reads_declared_peer_origins() -> None:
    table, _cfg = build_config(_spark_lobe_env(peers=True))
    assert dict(table.peer_origins) == {"multimodal": _THOR_ORIGIN}


def test_build_config_peer_origin_blank_is_unset() -> None:
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_ORIGIN="  "))
    assert dict(table.peer_origins) == {}


def test_build_config_peer_origin_trailing_slash_stripped() -> None:
    table, _cfg = build_config(_spark_lobe_env(MULTIMODAL_PEER_ORIGIN=_THOR_ORIGIN + "/"))
    assert dict(table.peer_origins) == {"multimodal": _THOR_ORIGIN}


def test_peer_origin_is_never_derived() -> None:
    # The #92 lesson: no env declaration, no origin — nothing is ever inferred
    # from hostnames/interfaces, even for a role that is clearly dropped.
    table, _cfg = build_config(_spark_lobe_env())
    assert "multimodal" in table.infeasible
    assert table.peer_origins.get("multimodal") is None


# ============================================================================
# Capabilities annotation (gateway payload + shared helper)
# ============================================================================


def test_capabilities_annotates_unhosted_role_with_peer_origin() -> None:
    env = _spark_lobe_env(peers=True)
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert payload["senses"]["hosted_by"] == _THOR_ORIGIN
    # Existing honesty flags untouched.
    assert payload["senses"]["feasible"] is False
    assert payload["senses"]["ready"] is False


def test_capabilities_never_annotates_hosted_roles() -> None:
    env = _spark_lobe_env(peers=True)
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    for role in ("cortex", "embedder", "reranker", "stt", "tts"):
        assert "hosted_by" not in payload[role], role


def test_peer_declared_for_a_hosted_role_is_ignored() -> None:
    # A referral says who hosts a role THIS box does not serve. A peer origin
    # declared for a locally-hosted role annotates nothing — the role is here.
    env = _spark_lobe_env(peers=True, PRIMARY_PEER_ORIGIN=_SPARK_ORIGIN)
    table, cfg = build_config(env)
    payload = S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    assert "hosted_by" not in payload["cortex"]


def test_annotate_peer_referrals_requires_declared_origin() -> None:
    # The helper itself: infeasible but undeclared → untouched.
    env = _spark_lobe_env()
    table, cfg = build_config(env)
    registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    payload = {role: dataclasses.asdict(registry[role]) for role in ROLES}
    annotate_peer_referrals(payload, table)
    for role in ROLES:
        assert "hosted_by" not in payload[role], role


# ============================================================================
# Byte-identity regression: zero peer config == the pre-change contract
# ============================================================================


def test_capabilities_bytes_identical_without_peer_config() -> None:
    # The pre-change /capabilities construction was literally
    # {role: dataclasses.asdict(registry[role])} — with no peer config the
    # payload must serialise to those exact bytes, nothing added.
    env = _spark_lobe_env()
    table, cfg = build_config(env)
    registry = build_role_registry(table, cfg, env=env, gateway_url=_GATEWAY_URL)
    expected = json.dumps({role: dataclasses.asdict(registry[role]) for role in ROLES})
    got = json.dumps(S.capabilities_payload(table, cfg, env=env, gateway_url=_GATEWAY_URL))
    assert got == expected
    assert "hosted_by" not in got


def test_role_infeasible_404_bytes_identical_without_peer_config() -> None:
    # The exact pre-change 404 body, byte for byte.
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


def test_v1_models_unaffected_by_peer_config() -> None:
    # /v1/models stays unchanged either way: it omits the unhosted role and
    # never carries a referral.
    with_peers, _ = build_config(_spark_lobe_env(peers=True))
    without, _ = build_config(_spark_lobe_env())
    ready = {"primary": True, "embed": True, "rerank": True}
    assert json.dumps(list_models_payload(with_peers, ready)) == json.dumps(
        list_models_payload(without, ready)
    )
    ids = {e["id"] for e in list_models_payload(with_peers, ready)["data"]}
    assert _SENSES_ID not in ids


# ============================================================================
# The 404 referral — and NO outbound connection, ever (handle_post seam)
# ============================================================================


@pytest.mark.parametrize("alias", ["senses", "multimodal", "normal"])
def test_404_carries_referral_and_dials_nothing(alias: str) -> None:
    table, cfg = build_config(_spark_lobe_env(peers=True))
    resp, calls = _post(table, cfg, alias)
    assert resp.status == 404
    assert calls == []  # the referral is an ANSWER, never a forward
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["code"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _THOR_ORIGIN
    assert _THOR_ORIGIN in body["error"]["message"]


def test_embed_request_for_unhosted_embedder_404s_with_referral_no_dial() -> None:
    env = _spark_lobe_env(EMBED_FEASIBLE="false", EMBED_PEER_ORIGIN=_THOR_ORIGIN)
    table, cfg = build_config(env)
    resp, calls = _post(table, cfg, _EMBED_ID, path="/v1/embeddings")
    assert resp.status == 404
    assert calls == []
    body = json.loads(resp.body)
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _THOR_ORIGIN


def test_hosted_role_still_routes_locally_with_peers_configured() -> None:
    # Peer config annotates honesty surfaces only — it never changes routing
    # for hosted roles.
    table, cfg = build_config(_spark_lobe_env(peers=True))
    resp, calls = _post(table, cfg, "cortex")
    assert resp.status == 200
    assert calls == ["primary"]


# ============================================================================
# Loopback: the real HTTP gateway, with a hard no-outbound guard
# ============================================================================


@pytest.fixture
def spark_lobe_gateway_with_peers(monkeypatch):
    """A real gateway serving a spark-lobe env with peers declared.

    ``S.open_upstream`` — the ONLY seam through which the gateway ever dials
    a backend or anything else — is replaced with a stub that records and
    fails, so any attempt to open an outbound connection is both visible and
    fatal to the test.
    """
    env = _spark_lobe_env(peers=True)
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
    try:
        yield f"http://{host}:{port}", outbound
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_integration_capabilities_carries_referral(spark_lobe_gateway_with_peers) -> None:
    url, outbound = spark_lobe_gateway_with_peers
    with urllib.request.urlopen(url + "/capabilities", timeout=5) as r:
        payload = json.load(r)
    assert payload["senses"]["hosted_by"] == _THOR_ORIGIN
    assert payload["senses"]["feasible"] is False
    assert "hosted_by" not in payload["cortex"]
    assert outbound == []


@pytest.mark.parametrize(
    "path, model",
    [
        ("/v1/chat/completions", "senses"),
        ("/v1/chat/completions", "multimodal"),
        ("/v1/chat/completions", "normal"),
    ],
)
def test_integration_unhosted_role_404_referral_no_outbound(
    spark_lobe_gateway_with_peers, path: str, model: str
) -> None:
    url, outbound = spark_lobe_gateway_with_peers
    req = urllib.request.Request(
        url + path,
        data=json.dumps({"model": model, "messages": []}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404
    body = json.loads(exc.value.read())
    assert body["error"]["type"] == "role_infeasible"
    assert body["error"]["hosted_by"] == _THOR_ORIGIN
    # The proof for acceptance criterion 2: the gateway answered the unhosted
    # role locally and NEVER opened an outbound connection.
    assert outbound == []


def test_integration_audio_request_unconfigured_no_outbound(
    spark_lobe_gateway_with_peers,
) -> None:
    # The audio lane on a box without the overlay: 404 locally, no forward.
    url, outbound = spark_lobe_gateway_with_peers
    req = urllib.request.Request(
        url + "/v1/audio/speech",
        data=json.dumps({"input": "hi", "voice": "x"}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 404
    assert outbound == []


# ============================================================================
# The CLI verb (offline fallback path — gateway mode renders verbatim anyway)
# ============================================================================


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def test_cli_capabilities_offline_annotates_referral(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_FEASIBLE", "false")
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGIN", _SPARK_ORIGIN)
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cortex"]["hosted_by"] == _SPARK_ORIGIN
    assert payload["cortex"]["feasible"] is False
    for role in ("senses", "embedder", "reranker", "stt", "tts"):
        assert "hosted_by" not in payload[role], role


def test_cli_capabilities_offline_bytes_identical_without_peer_config(tmp_path, capsys) -> None:
    # With no peer config the CLI's JSON payload carries EXACTLY the RoleInfo
    # field set per role — no hosted_by key anywhere (byte-identity with the
    # pre-change contract at the payload level).
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_FEASIBLE", "false")
    rc = main(["capabilities", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "hosted_by" not in out
    from lobes.roles import RoleInfo

    fields = {f.name for f in dataclasses.fields(RoleInfo)}
    payload = json.loads(out)
    for role in ROLES:
        assert set(payload[role]) == fields, role


def test_cli_capabilities_table_shows_referral(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_FEASIBLE", "false")
    _env.set_env(tmp_path / _compose.ENV_FILE, "PRIMARY_PEER_ORIGIN", _SPARK_ORIGIN)
    rc = main(["capabilities", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "infeasible on this machine" in out
    assert _SPARK_ORIGIN in out
