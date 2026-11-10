"""Gateway evidence for the site: host memory only (t21, amended by deviation d2).

* ``/health`` carries ``memory`` read from ``/proc/meminfo`` (omitted when
  unreadable).
* Deviation d2 removed the hand-defined split/tunnel declarations
  (``<PREFIX>_SPANS``, ``<PREFIX>_WORLD_SIZE``, ``LOBES_TUNNEL_PUBLIC_URL``):
  setting them changes nothing on ``/capabilities`` or ``/health``. The site
  reads split brains and the hub from the running containers instead.
"""

from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from lobes import __version__
from lobes.gateway import _evidence as E
from lobes.gateway import server as S
from lobes.gateway._config import build_config
from lobes.gateway._mesh_routing import RoutingSnapshot

ORIGIN = "http://spark:8001"
BASE_ENV = {
    "PRIMARY_BASE_URL": "http://vllm-primary:8000",
    "PRIMARY_SERVED_NAME": "zai/GLM-5.3-Flash",
    "PRIMARY_MAX_MODEL_LEN": "131072",
}
RETIRED = {
    "PRIMARY_SPANS": "spark,spark2",
    "PRIMARY_WORLD_SIZE": "2",
    "LOBES_TUNNEL_PUBLIC_URL": "https://lobes.culture.dev",
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


# ------------------------------------------------- d2: declarations retired


def test_retired_split_keys_do_not_change_capabilities():
    assert _payload({**BASE_ENV, **RETIRED}) == _payload(BASE_ENV)
    assert "spans" not in _payload({**BASE_ENV, **RETIRED})["cortex"]
    assert "world_size" not in _payload({**BASE_ENV, **RETIRED})["cortex"]


def test_retired_api_is_gone():
    for name in ("split_facts", "annotate_split", "SPLIT_ROLE_PREFIX", "TUNNEL_ENV"):
        assert not hasattr(E, name), name


def test_health_never_carries_a_declared_tunnel(tmp_path):
    body = E.health_payload(version="x", meminfo=tmp_path / "missing")
    assert "tunnel" not in body


# ------------------------------------------------------------------ /health

_MEMINFO = "MemTotal:       131072000 kB\nMemFree:  1000 kB\nMemAvailable:    65536000 kB\n"


def test_health_payload_unchanged_when_nothing_measured(tmp_path):
    body = E.health_payload(version="1.2.3", meminfo=tmp_path / "missing")
    assert body == {"status": "ok", "service": "model-gear-gateway", "version": "1.2.3"}


def test_health_payload_memory_from_meminfo(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text(_MEMINFO)
    body = E.health_payload(version="1.2.3", meminfo=mi)
    assert body["memory"] == {
        "total_bytes": 131072000 * 1024,
        "available_bytes": 65536000 * 1024,
    }
    assert "tunnel" not in body


def test_health_payload_memory_omitted_when_meminfo_incomplete(tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text("MemTotal: 100 kB\n")
    assert "memory" not in E.health_payload(version="x", meminfo=mi)


def test_health_route_serves_the_evidence(monkeypatch, tmp_path):
    mi = tmp_path / "meminfo"
    mi.write_text(_MEMINFO)
    monkeypatch.setattr(E, "MEMINFO", mi)
    monkeypatch.setenv("LOBES_TUNNEL_PUBLIC_URL", "https://lobes.culture.dev")  # retired: inert
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
    assert body["status"] == "ok"
    assert body["version"] == __version__
    assert body["memory"]["total_bytes"] == 131072000 * 1024
    assert "tunnel" not in body
