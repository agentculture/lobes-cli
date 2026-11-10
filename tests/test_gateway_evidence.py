"""Gateway evidence for the site (t21, deviation d1): split, memory, tunnel.

* ``/capabilities`` carries per-role ``spans`` / ``world_size`` only when the
  operator typed ``<PREFIX>_SPANS`` / ``<PREFIX>_WORLD_SIZE`` (#92: never
  derived). Unset -> the payload is byte-identical to before.
* The mesh fingerprint (pool agreement / verification) never sees them.
* ``/health`` carries ``memory`` read from ``/proc/meminfo`` (omitted when
  unreadable) and ``tunnel`` only when ``LOBES_TUNNEL_PUBLIC_URL`` is set.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from lobes import __version__
from lobes.gateway import _evidence as E
from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_config import MeshConfig
from lobes.gateway._mesh_routes import announcement_from_capabilities
from lobes.gateway._mesh_routing import RoutingSnapshot, verify_member_roles
from lobes.gateway._mesh_wire import capability_entries, decode, encode

ORIGIN = "http://spark:8001"
BASE_ENV = {
    "PRIMARY_BASE_URL": "http://vllm-primary:8000",
    "PRIMARY_SERVED_NAME": "zai/GLM-5.3-Flash",
    "PRIMARY_MAX_MODEL_LEN": "131072",
}


def _payload(env: dict[str, str]) -> dict:
    table, cfg = build_config(env)
    ready = {b.name: True for b in table.backends}
    return S.capabilities_payload(
        table,
        cfg,
        env,
        gateway_url=ORIGIN,
        backend_ready=ready,
        mesh_snapshot=RoutingSnapshot(members=(), announcements=()),
    )


# ------------------------------------------------------------- split facts


def test_split_facts_empty_when_unset():
    assert E.split_facts(BASE_ENV) == {}


def test_split_facts_reads_operator_typed_keys():
    env = {**BASE_ENV, "PRIMARY_SPANS": " spark , spark2 ", "PRIMARY_WORLD_SIZE": "2"}
    assert E.split_facts(env) == {"cortex": {"spans": ["spark", "spark2"], "world_size": 2}}


def test_split_facts_prefix_table_covers_generate_roles_only():
    assert E.SPLIT_ROLE_PREFIX == {
        "cortex": "PRIMARY",
        "senses": "MULTIMODAL",
        "muse": "MUSE",
        "worker": "WORKER",
        "associate": "ASSOCIATE",
        "hand": "HAND",
    }
    env = {"WORKER_WORLD_SIZE": "4", "EMBED_WORLD_SIZE": "2", "EMBED_SPANS": "a,b"}
    assert E.split_facts(env) == {"worker": {"world_size": 4}}


@pytest.mark.parametrize(
    "key,value",
    [
        ("PRIMARY_SPANS", "spark"),  # one name is not a split
        ("PRIMARY_SPANS", "spark,spark"),  # duplicates collapse to one
        ("PRIMARY_SPANS", "spark,bad name"),  # outside the mesh-name rule
        ("PRIMARY_SPANS", "spark,,spark2,"),  # empty entries are malformed
        ("PRIMARY_WORLD_SIZE", "0"),
        ("PRIMARY_WORLD_SIZE", "-2"),
        ("PRIMARY_WORLD_SIZE", "two"),
        ("PRIMARY_WORLD_SIZE", "2.5"),
    ],
)
def test_malformed_value_is_ignored_with_a_log(key, value, capsys):
    E._WARNED.clear()
    assert E.split_facts({**BASE_ENV, key: value}) == {}
    err = capsys.readouterr().err
    assert key in err and "ignored" in err


def test_malformed_value_logs_once_not_per_request(capsys):
    E._WARNED.clear()
    env = {**BASE_ENV, "PRIMARY_WORLD_SIZE": "zero"}
    E.split_facts(env)
    E.split_facts(env)
    assert capsys.readouterr().err.count("PRIMARY_WORLD_SIZE") == 1


def test_one_malformed_key_keeps_the_other():
    E._WARNED.clear()
    env = {**BASE_ENV, "PRIMARY_SPANS": "spark,spark2", "PRIMARY_WORLD_SIZE": "x"}
    assert E.split_facts(env) == {"cortex": {"spans": ["spark", "spark2"]}}


# ------------------------------------------------------------- /capabilities


def test_capabilities_byte_identical_when_unset():
    plain = json.dumps(_payload(BASE_ENV), sort_keys=True)
    blanks = {**BASE_ENV, "PRIMARY_SPANS": "", "PRIMARY_WORLD_SIZE": " "}
    assert json.dumps(_payload(blanks), sort_keys=True) == plain
    assert "spans" not in _payload(BASE_ENV)["cortex"]
    assert "world_size" not in _payload(BASE_ENV)["cortex"]


def test_capabilities_carries_spans_and_world_size():
    env = {**BASE_ENV, "PRIMARY_SPANS": "spark,spark2", "PRIMARY_WORLD_SIZE": "2"}
    payload = _payload(env)
    assert payload["cortex"]["spans"] == ["spark", "spark2"]
    assert payload["cortex"]["world_size"] == 2
    # only the declared role moves; every other key of every role is unchanged
    plain = _payload(BASE_ENV)
    stripped = {k: v for k, v in payload["cortex"].items() if k not in ("spans", "world_size")}
    assert stripped == plain["cortex"]
    assert {k: v for k, v in payload.items() if k != "cortex"} == {
        k: v for k, v in plain.items() if k != "cortex"
    }


def _mesh_cfg() -> MeshConfig:
    return MeshConfig(
        enabled=True,
        key="sk-mesh",
        name="spark",
        seeds=(),
        heartbeat_s=60,
        missed_max=3,
        ledger_path=None,
    )


def test_fingerprint_and_announcement_ignore_split_keys():
    env = {**BASE_ENV, "PRIMARY_SPANS": "spark,spark2", "PRIMARY_WORLD_SIZE": "2"}
    plain, split = _payload(BASE_ENV), _payload(env)
    assert split["cortex"]["fingerprint"] == plain["cortex"]["fingerprint"]
    a_plain = announcement_from_capabilities(_mesh_cfg(), plain, self_origin=ORIGIN)
    a_split = announcement_from_capabilities(_mesh_cfg(), split, self_origin=ORIGIN)
    assert encode(a_split) == encode(a_plain)
    assert b"spans" not in encode(a_split) and b"world_size" not in encode(a_split)


def test_verification_passes_with_split_keys_on_the_probed_side():
    env = {**BASE_ENV, "PRIMARY_SPANS": "spark,spark2", "PRIMARY_WORLD_SIZE": "2"}
    split = _payload(env)
    # the peer ANNOUNCED without the keys (e.g. pre-t21) and is PROBED with them
    ann = decode(
        encode(announcement_from_capabilities(_mesh_cfg(), _payload(BASE_ENV), self_origin=ORIGIN))
    )
    probed = {k: dict(v) for k, v in capability_entries(split).items() if isinstance(v, dict)}
    assert "cortex" in verify_member_roles(ann, probed)


# ------------------------------------------------------------------ /health

_MEMINFO = "MemTotal:       131072000 kB\nMemFree:  1000 kB\nMemAvailable:    65536000 kB\n"


def test_health_payload_unchanged_when_nothing_measured_or_declared(tmp_path):
    body = E.health_payload({}, version="1.2.3", meminfo=tmp_path / "missing")
    assert body == {"status": "ok", "service": "model-gear-gateway", "version": "1.2.3"}


def test_health_payload_memory_from_meminfo(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text(_MEMINFO)
    body = E.health_payload({}, version="1.2.3", meminfo=mi)
    assert body["memory"] == {
        "total_bytes": 131072000 * 1024,
        "available_bytes": 65536000 * 1024,
    }
    assert "tunnel" not in body


def test_health_payload_memory_omitted_when_meminfo_incomplete(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 100 kB\n")
    assert "memory" not in E.health_payload({}, version="x", meminfo=mi)


def test_health_payload_tunnel_only_when_declared(tmp_path):
    missing = tmp_path / "missing"
    assert "tunnel" not in E.health_payload(
        {"LOBES_TUNNEL_PUBLIC_URL": " "}, version="x", meminfo=missing
    )
    body = E.health_payload(
        {"LOBES_TUNNEL_PUBLIC_URL": " https://lobes.culture.dev/ "}, version="x", meminfo=missing
    )
    assert body["tunnel"] == {"public_url": "https://lobes.culture.dev", "declared": True}


def test_health_route_serves_the_evidence(monkeypatch, tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text(_MEMINFO)
    monkeypatch.setattr(E, "MEMINFO", mi)
    monkeypatch.setenv("LOBES_TUNNEL_PUBLIC_URL", "https://lobes.culture.dev")
    table, cfg = build_config({"PRIMARY_SERVED_NAME": "P"})
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S._make_handler(table, cfg))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        host, port = httpd.server_address
        with urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5) as r:
            body = json.load(r)
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["memory"]["total_bytes"] == 131072000 * 1024
    assert body["tunnel"] == {"public_url": "https://lobes.culture.dev", "declared": True}
