"""lobes site: the local server and its /api (t17).

Every data source is injected; no test touches a real network or remote host.
Servers bind port 0 on loopback and are stopped in the same test.
"""

from __future__ import annotations

import http.client
import json
from pathlib import Path

import pytest

from lobes.cli import main
from lobes.rollout._journal import STEPS, Journal
from lobes.rollout._observe import MemberObservation
from lobes.rollout._residents import DETECTED, NEEDS_YOU, Field, Residents
from lobes.site import server as site

SECRET = "sk-mesh-very-secret-0001"
GW_SECRET = "gw-secret-key-0002"

LOCAL = "http://127.0.0.1:8001"


def _role(model, *, ready=True, feasible=True, **extra):
    return {"model": model, "runtime": "vllm", "ready": ready, "feasible": feasible, **extra}


def _roster(*members):
    return [
        {"name": n, "origin": o, "verified": True, "roles": roles, "probed": True}
        for n, o, roles in members
    ]


ROSTER = _roster(
    ("spark", "http://spark:8001", ["cortex"]),
    ("spark2", "http://spark2:8000", ["cortex", "stt-en"]),
    ("thor", "http://thor:8000", ["reranker", "worker"]),
    ("orin", "http://orin:8000", ["reranker", "embedder"]),
)


def _caps():
    return {
        LOCAL: {
            # reaches cortex only through the mesh: not a lobe here
            "cortex": _role("GLM-5.3-Flash", proxied=True, hosted_by="http://spark:8001"),
            "tts": _role("ResembleAI/chatterbox", proxied=True),
        },
        "http://spark:8001": {"cortex": _role("GLM-5.3-Flash", world_size=2)},
        "http://spark2:8000": {
            "cortex": _role("GLM-5.3-Flash", world_size=2),
            "stt-en": _role("nvidia/parakeet-tdt-0.6b-v2", runtime="parakeet"),
        },
        "http://thor:8000": {
            "reranker": _role("Qwen/Qwen3-Reranker-0.6B"),
            "worker": _role("Nemotron-Super", world_size=1),
            "muse": _role("gemma", feasible=False),
        },
        "http://orin:8000": {
            "reranker": _role("Qwen/Qwen3-Reranker-0.6B"),
            "embedder": _role("Qwen/Qwen3-Embedding-0.6B"),
        },
    }


VERSIONS = {
    LOCAL: "0.94.0",
    "http://spark:8001": "0.92.0.dev694",
    "http://spark2:8000": "0.94.0",
    "http://thor:8000": "0.92.0.dev694",
    "http://orin:8000": "0.94.0",
}


def make_observe(caps=None, versions=None, roster=None, down=()):
    caps = _caps() if caps is None else caps
    versions = VERSIONS if versions is None else versions
    roster = ROSTER if roster is None else roster
    calls = []

    def observe(origin):
        calls.append(origin)
        if origin in down:
            return MemberObservation(origin=origin)
        return MemberObservation(
            origin=origin,
            version=versions.get(origin),
            capabilities=caps.get(origin, {}),
            roster=roster if origin == LOCAL else [],
            reachable=True,
        )

    observe.calls = calls
    return observe


def _f(v):
    return Field(v, DETECTED) if v is not None else Field(None, NEEDS_YOU)


def _entry(ssh, user, port, path, key):
    return {
        "ssh": _f(ssh),
        "user": _f(user),
        "port": _f(port),
        "lobes_path": _f(path),
        "gateway_key": _f(key),
    }


def make_residents():
    return Residents(
        residents={
            "cultureos": _entry("local", "arduino", 8001, "/home/a/.local/bin/lobes", True),
            "spark": _entry("spark", "spark", 8001, "/home/s/.local/bin/lobes", True),
            "spark2": _entry("spark2", "spark2", 8000, "/x/lobes", True),
            "thor": _entry("thor", "thor", 8000, "/x/lobes", False),
        },
        guests={"orin": _entry("orin", None, 8000, None, True)},
    )


def make_sources(tmp_path, **over):
    journal = Journal(tmp_path)
    kw = dict(
        local_origin=LOCAL,
        local_name="cultureos",
        observe=make_observe(),
        residents=make_residents,
        journal=lambda: journal,
        export_listing=lambda role, member: {
            "role": role,
            "member": member,
            "status": "built",
            "files": [f"{member}/{role}/Dockerfile"],
        },
        memory=lambda name, local, health: (
            {"total_bytes": 128 * 2**30, "available_bytes": 32 * 2**30} if local else None
        ),
        tunnel=lambda name, local, health: None,
        secrets=(SECRET, GW_SECRET),
    )
    kw.update(over)
    return site.Sources(**kw)


@pytest.fixture
def serve(tmp_path):
    started = []

    def _start(sources=None, web_root=None):
        srv = site.SiteServer(
            sources or make_sources(tmp_path), host="127.0.0.1", port=0, web_root=web_root
        )
        srv.start()
        started.append(srv)
        return srv

    yield _start
    for s in started:
        s.stop()


def _get(srv, path, method="GET"):
    host, port = srv.address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=5)
    try:
        conn.request(method, path)
        r = conn.getresponse()
        return r.status, dict(r.getheaders()), r.read()
    finally:
        conn.close()


def _json(srv, path):
    code, _, body = _get(srv, path)
    return code, json.loads(body)


# ------------------------------------------------------------------ static


def test_index_is_served_with_csp(serve):
    code, headers, body = _get(serve(), "/")
    assert code == 200
    assert headers["Content-Type"].startswith("text/html")
    assert "Content-Security-Policy" in headers
    assert b"<html" in body.lower()


def test_vendor_serves_allow_listed_file(serve, tmp_path):
    root = tmp_path / "web"
    (root / "vendor").mkdir(parents=True)
    (root / "index.html").write_text("<html></html>")
    (root / "vendor" / "lato-OFL.txt").write_text("OFL")
    code, headers, body = _get(serve(web_root=root), "/vendor/lato-OFL.txt")
    assert code == 200 and body == b"OFL"
    assert headers["Content-Type"].startswith("text/plain")


@pytest.mark.parametrize(
    "path",
    [
        "/vendor/../index.html",
        "/vendor/..%2findex.html",
        "/vendor/%2e%2e/secret.txt",
        "/vendor/sub/lato-OFL.txt",
        "/vendor/..",
        "/vendor/secret.py",
        "/vendor/",
    ],
)
def test_vendor_path_traversal_refused(serve, tmp_path, path):
    root = tmp_path / "web"
    (root / "vendor" / "sub").mkdir(parents=True)
    (root / "index.html").write_text("<html></html>")
    (root / "vendor" / "sub" / "lato-OFL.txt").write_text("nested")
    (root / "vendor" / "secret.py").write_text("x")
    (tmp_path / "secret.txt").write_text("outside")
    code, _, body = _get(serve(web_root=root), path)
    assert code == 404
    assert b"outside" not in body and b"nested" not in body


def test_other_methods_refused_and_unknown_404(serve):
    srv = serve()
    assert _get(srv, "/api/mesh", method="POST")[0] == 405
    assert _get(srv, "/api/nope")[0] == 404


def test_default_bind_is_loopback():
    assert site.DEFAULT_HOST == "127.0.0.1"


# ------------------------------------------------------------------ /api/mesh


def test_mesh_members_versions_roles_and_lobes(serve):
    code, data = _json(serve(), "/api/mesh")
    assert code == 200
    names = [m["name"] for m in data["members"]]
    assert names == ["cultureos", "spark", "spark2", "thor", "orin"]
    by = {m["name"]: m for m in data["members"]}
    assert by["cultureos"]["local"] is True and by["spark"]["local"] is False
    assert by["spark"]["version"] == "0.92.0.dev694"
    assert data["versions"] == {
        "0.94.0": ["cultureos", "spark2", "orin"],
        "0.92.0.dev694": ["spark", "thor"],
    }
    assert by["thor"]["roles"] == ["reranker", "worker"]
    # proxied (mesh-reached) and infeasible roles are not lobes on that member
    assert by["cultureos"]["lobes"] == []
    thor_lobes = {lobe["role"]: lobe for lobe in by["thor"]["lobes"]}
    assert set(thor_lobes) == {"reranker", "worker"}
    assert thor_lobes["reranker"]["model"] == "Qwen/Qwen3-Reranker-0.6B"
    spark2 = {lobe["role"]: lobe["model"] for lobe in by["spark2"]["lobes"]}
    assert spark2 == {"cortex": "GLM-5.3-Flash", "stt-en": "nvidia/parakeet-tdt-0.6b-v2"}


def test_mesh_residency_from_residents_file(serve):
    _, data = _json(serve(), "/api/mesh")
    by = {m["name"]: m for m in data["members"]}
    assert by["spark"]["residency"] == "resident"
    assert by["orin"]["residency"] == "guest"


def test_mesh_memory_only_from_a_source(serve):
    _, data = _json(serve(), "/api/mesh")
    by = {m["name"]: m for m in data["members"]}
    assert by["cultureos"]["memory"] == {
        "total_bytes": 128 * 2**30,
        "available_bytes": 32 * 2**30,
        "used_bytes": 96 * 2**30,
    }
    assert by["thor"]["memory"] is None  # no evidence -> unknown, never a guess


def test_split_brain_from_world_size_evidence(serve):
    _, data = _json(serve(), "/api/mesh")
    assert len(data["split_brains"]) == 1
    sb = data["split_brains"][0]
    assert sb["model"] == "GLM-5.3-Flash"
    assert sb["members"] == ["spark", "spark2"]
    assert "world_size" in sb["evidence"]
    by = {m["name"]: m for m in data["members"]}
    assert {lobe["shape"] for lobe in by["spark"]["lobes"]} == {"split"}


def test_pair_is_one_model_pooled_on_several(serve):
    _, data = _json(serve(), "/api/mesh")
    assert [(p["model"], p["members"]) for p in data["pairs"]] == [
        ("Qwen/Qwen3-Reranker-0.6B", ["thor", "orin"])
    ]
    # the split model is not also reported as a pair
    assert all(p["model"] != "GLM-5.3-Flash" for p in data["pairs"])
    by = {m["name"]: m for m in data["members"]}
    shapes = {lobe["role"]: lobe["shape"] for lobe in by["orin"]["lobes"]}
    assert shapes == {"reranker": "pair", "embedder": "single"}


def test_no_split_evidence_means_pair_not_split(serve, tmp_path):
    caps = _caps()
    for o in ("http://spark:8001", "http://spark2:8000"):
        caps[o]["cortex"].pop("world_size")
    _, data = _json(serve(make_sources(tmp_path, observe=make_observe(caps=caps))), "/api/mesh")
    assert data["split_brains"] == []
    assert ("GLM-5.3-Flash", ["spark", "spark2"]) in [
        (p["model"], p["members"]) for p in data["pairs"]
    ]


def test_world_size_on_one_member_alone_is_single(serve, tmp_path):
    caps = _caps()
    caps["http://spark2:8000"].pop("cortex")
    _, data = _json(serve(make_sources(tmp_path, observe=make_observe(caps=caps))), "/api/mesh")
    assert data["split_brains"] == []
    by = {m["name"]: m for m in data["members"]}
    assert by["spark"]["lobes"][0]["shape"] == "single"


def test_spans_evidence_names_the_members(serve, tmp_path):
    caps = _caps()
    caps["http://spark:8001"]["cortex"] = _role("GLM-5.3-Flash", spans=["spark", "spark2"])
    caps["http://spark2:8000"]["cortex"] = _role("GLM-5.3-Flash")
    _, data = _json(serve(make_sources(tmp_path, observe=make_observe(caps=caps))), "/api/mesh")
    assert [s["members"] for s in data["split_brains"]] == [["spark", "spark2"]]
    assert data["pairs"] == [
        p for p in data["pairs"] if p["model"] != "GLM-5.3-Flash"
    ]  # not double-counted


def test_hub_only_when_evidenced(serve, tmp_path):
    _, data = _json(serve(), "/api/mesh")
    assert data["hub"] is None
    ev = make_sources(
        tmp_path,
        tunnel=lambda name, local, health: "cloudflared container running" if local else None,
    )
    _, data = _json(serve(ev), "/api/mesh")
    assert data["hub"] == {"member": "cultureos", "evidence": "cloudflared container running"}


def test_unreachable_member_has_no_lobes(serve, tmp_path):
    src = make_sources(tmp_path, observe=make_observe(down=("http://thor:8000",)))
    _, data = _json(serve(src), "/api/mesh")
    by = {m["name"]: m for m in data["members"]}
    assert by["thor"]["reachable"] is False
    assert by["thor"]["lobes"] == [] and by["thor"]["version"] is None
    assert by["thor"]["roles"] == ["reranker", "worker"]  # roster still says so
    assert data["pairs"] == []


def test_mesh_data_is_live_not_hard_coded(serve, tmp_path):
    roster = _roster(("box-a", "http://a:1", ["hand"]))
    caps = {LOCAL: {}, "http://a:1": {"hand": _role("LiquidAI/LFM2.5-1.2B-Instruct")}}
    src = make_sources(
        tmp_path,
        local_name="box-z",
        observe=make_observe(caps=caps, versions={"http://a:1": "9.9.9"}, roster=roster),
    )
    _, data = _json(serve(src), "/api/mesh")
    assert [m["name"] for m in data["members"]] == ["box-z", "box-a"]
    assert data["members"][1]["lobes"][0]["model"] == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert data["versions"] == {"9.9.9": ["box-a"], "unknown": ["box-z"]}


# ------------------------------------------------------------ rollout/residents


def test_api_rollout_projects_the_journal(serve, tmp_path):
    j = Journal(tmp_path)
    j.append("thor", "verify", "ok", target="0.94.0", current="0.94.0")
    j.append("spark", "rebuild_gateway", "ok", target="0.94.0", current="0.92.0")
    code, data = _json(serve(make_sources(tmp_path, journal=lambda: j)), "/api/rollout")
    assert code == 200
    by = {m["member"]: m for m in data["members"]}
    assert by["thor"]["state"] == "done" and by["thor"]["done"] is True
    assert by["spark"]["state"] == "in_progress" and by["spark"]["last_step"] == "rebuild_gateway"
    assert data["resume"] == {"member": "spark", "step": "wait_health"}


def test_api_rollout_empty_journal(serve):
    code, data = _json(serve(), "/api/rollout")
    assert code == 200 and data["members"] == [] and data["resume"] is None
    assert data["steps"] == list(STEPS)


def test_api_rollout_names_the_rollback_for_a_failed_member(serve, tmp_path):
    j = Journal(tmp_path)
    j.append("thor", "verify", "ok", target="0.94.0", current="0.94.0")
    j.append("spark", "verify", "failed", target="0.94.0", current="0.94.0")
    code, data = _json(serve(make_sources(tmp_path, journal=lambda: j)), "/api/rollout")
    by = {m["member"]: m for m in data["members"]}
    assert by["spark"]["rollback_command"] == "lobes rollout rollback --member spark --apply"
    assert by["thor"]["rollback_command"] is None


def test_api_residents(serve):
    code, data = _json(serve(), "/api/residents")
    assert code == 200
    res = {r["name"]: r for r in data["residents"]}
    assert res["thor"]["fields"]["gateway_key"] == {"value": False, "state": "detected"}
    assert res["spark"]["needs_you"] == []
    guests = {g["name"]: g for g in data["guests"]}
    assert guests["orin"]["fields"]["user"] == {"value": None, "state": "needs_you"}
    assert guests["orin"]["needs_you"] == ["user", "lobes_path"]


# ------------------------------------------------------------------ export


def test_api_lobe_export_lists_the_bundle(serve, tmp_path):
    seen = []

    def listing(role, member):
        seen.append((role, member))
        return {"role": role, "member": member, "status": "built", "files": ["a/b"]}

    srv = serve(make_sources(tmp_path, export_listing=listing))
    code, data = _json(srv, "/api/lobe-export?role=stt-en&member=spark2")
    assert code == 200
    assert seen == [("stt-en", "spark2")]
    assert data["files"] == ["a/b"]
    assert data["cli"] == "lobes lobe export stt-en --member spark2"


@pytest.mark.parametrize(
    "query,code",
    [
        ("", 400),
        ("?role=stt-en", 400),
        ("?role=stt-en&member=../etc", 400),
        ("?role=a%20b&member=spark", 400),
        ("?role=stt-en&member=nobody", 404),
    ],
)
def test_api_lobe_export_validates(serve, query, code):
    assert _get(serve(), "/api/lobe-export" + query)[0] == code


def test_api_lobe_export_failure_is_502_without_traceback(serve, tmp_path):
    def boom(role, member):
        raise site.ExportUnavailable(f"ssh failed: Authorization: Bearer {SECRET}")

    srv = serve(make_sources(tmp_path, export_listing=boom))
    code, _, body = _get(srv, "/api/lobe-export?role=cortex&member=spark")
    assert code == 502
    assert SECRET.encode() not in body and b"Traceback" not in body


# ------------------------------------------------------------------ secrets


def test_no_secret_values_in_any_api_response(serve, tmp_path):
    caps = _caps()
    caps["http://thor:8000"]["worker"]["note"] = f"GATEWAY_API_KEY={GW_SECRET}"
    caps["http://orin:8000"]["embedder"]["endpoint"] = f"http://x/?k={SECRET}"
    j = Journal(tmp_path)
    j.append("thor", "install_cli", "failed", detail=f"token {SECRET}")

    def listing(role, member):
        return {"role": role, "member": member, "files": [], "notes": [f"LOBES_MESH_KEY={SECRET}"]}

    src = make_sources(
        tmp_path, observe=make_observe(caps=caps), journal=lambda: j, export_listing=listing
    )
    srv = serve(src)
    for path in (
        "/api/mesh",
        "/api/rollout",
        "/api/residents",
        "/api/lobe-export?role=x&member=spark",
    ):
        code, _, body = _get(srv, path)
        assert code == 200, path
        assert SECRET.encode() not in body and GW_SECRET.encode() not in body, path


# ------------------------------------------------------------- default sources


def test_default_sources_read_keys_from_env_and_resolve_local_origin(tmp_path):
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    (tmp_path / ".env").write_text(
        f"VLLM_PORT=8001\nLOBES_MESH_NAME=cultureos\nLOBES_MESH_KEY={SECRET}\n"
        f"GATEWAY_API_KEY={GW_SECRET}\n"
    )
    src = site.default_sources(tmp_path)
    assert src.local_origin == "http://127.0.0.1:8001"
    assert src.local_name == "cultureos"
    assert SECRET in src.secrets and GW_SECRET in src.secrets


def test_default_observe_uses_only_get(tmp_path):
    seen = []

    class Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b"{}"

    def opener(req, timeout):
        seen.append(req.get_method())
        return Resp()

    (tmp_path / ".env").write_text(f"LOBES_MESH_KEY={SECRET}\n")
    src = site.default_sources(tmp_path, opener=opener)
    src.observe("http://127.0.0.1:1")
    assert seen and set(seen) == {"GET"}


# ------------------------------------------------------------------ CLI


def _tree(p: Path):
    return sorted(str(x.relative_to(p)) for x in p.rglob("*"))


def test_cli_dry_run_prints_url_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    before = _tree(tmp_path)
    rc = main(["site", "--port", "9555"])
    assert rc == 0
    out = capsys.readouterr()
    assert "http://127.0.0.1:9555/" in out.out
    assert "--apply" in out.out + out.err
    assert _tree(tmp_path) == before


def test_cli_dry_run_json(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    rc = main(["site", "--json"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dry_run"] is True
    assert data["url"] == f"http://127.0.0.1:{site.DEFAULT_PORT}/"
    assert data["host"] == "127.0.0.1"
    assert "/api/mesh" in data["routes"]


def test_cli_apply_starts_the_server(tmp_path, capsys, monkeypatch):
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    (tmp_path / "docker-compose.yml").write_text("services: {}\n")
    ran = []

    def fake_run(srv):
        ran.append(srv.address)

    monkeypatch.setattr(site, "run_until_interrupted", fake_run)
    rc = main(["site", "--port", "0", "--apply"])
    assert rc == 0
    assert ran and ran[0][0] == "127.0.0.1"
    assert "http://127.0.0.1:" in capsys.readouterr().out


def test_cli_works_without_a_deployment(tmp_path, capsys):
    rc = main(["site", "--json"])  # conftest points ~/.lobes at an empty dir
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True


# ------------------------------------------------- t21: advertised evidence


def test_declared_hub_is_labelled_declared(serve, tmp_path):
    ev = make_sources(
        tmp_path,
        tunnel=lambda name, local, health: (
            {"evidence": "declared: lobes.culture.dev", "declared": True} if local else None
        ),
    )
    _, data = _json(serve(ev), "/api/mesh")
    assert data["hub"] == {
        "member": "cultureos",
        "evidence": "declared: lobes.culture.dev",
        "declared": True,
    }


class _GatewayOpener:
    """Serves fixture gateway JSON per origin; counts every request."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append(req.full_url)
        origin, _, path = req.full_url.partition("://")[2].partition("/")
        body = self.routes.get(f"http://{origin}", {}).get("/" + path)
        if body is None:
            raise OSError("unreachable")
        raw = json.dumps(body).encode()

        class R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return raw

        return R()


def _spark_caps():
    """spark's /capabilities as the real gateway renders it with t21's keys."""
    from lobes.gateway import server as gw
    from lobes.gateway._config import build_config
    from lobes.gateway._mesh_routing import RoutingSnapshot

    env = {
        "PRIMARY_BASE_URL": "http://tensorfold:8000",
        "PRIMARY_SERVED_NAME": "zai/GLM-5.3-Flash",
        "PRIMARY_SPANS": "spark,spark2",
        "PRIMARY_WORLD_SIZE": "2",
    }
    table, cfg = build_config(env)
    return gw.capabilities_payload(
        table,
        cfg,
        env,
        gateway_url="http://spark:8001",
        backend_ready={b.name: True for b in table.backends},
        mesh_snapshot=RoutingSnapshot(members=(), announcements=()),
    )


def _fleet_routes():
    from lobes.gateway import _evidence

    mem = {"total_bytes": 128 * 2**30, "available_bytes": 40 * 2**30}

    def health(version, **extra):
        return {"status": "ok", "service": "model-gear-gateway", "version": version, **extra}

    return {
        "http://127.0.0.1:8001": {
            "/health": _evidence.health_payload(
                {"LOBES_TUNNEL_PUBLIC_URL": "https://lobes.culture.dev"},
                version="0.95.0",
                meminfo=Path("/nonexistent/meminfo"),
            ),
            "/capabilities": {},
            "/mesh/roster": {
                "members": _roster(
                    ("spark", "http://spark:8001", ["cortex"]),
                    ("spark2", "http://spark2:8000", ["stt-en"]),
                    ("thor", "http://thor:8000", ["reranker"]),
                )
            },
        },
        "http://spark:8001": {
            "/health": health("0.95.0", memory=mem),
            "/capabilities": _spark_caps(),
            "/mesh/roster": {"members": []},
        },
        "http://spark2:8000": {
            "/health": health("0.95.0", memory={"total_bytes": 10, "available_bytes": 20}),
            "/capabilities": {"stt-en": _role("nvidia/parakeet-tdt-0.6b-v2")},
            "/mesh/roster": {"members": []},
        },
        "http://thor:8000": {
            "/health": health("0.94.0"),
            "/capabilities": {"reranker": _role("Qwen/Qwen3-Reranker-0.6B")},
            "/mesh/roster": {"members": []},
        },
    }


def _fleet_sources(tmp_path, monkeypatch, routes, *, local_tunnel=None, local_mem=None):
    (tmp_path / ".env").write_text("VLLM_PORT=8001\nLOBES_MESH_NAME=core\n")
    monkeypatch.setattr(site, "_local_tunnel", lambda base: local_tunnel)
    monkeypatch.setattr(site, "_read_meminfo", lambda path=None: local_mem)
    opener = _GatewayOpener(routes)
    return site.default_sources(tmp_path, opener=opener), opener


def test_gateway_shape_end_to_end_split_memory_and_declared_hub(tmp_path, monkeypatch):
    src, opener = _fleet_sources(tmp_path, monkeypatch, _fleet_routes())
    doc = site.build_mesh(src)
    assert [(s["model"], s["members"]) for s in doc["split_brains"]] == [
        ("zai/GLM-5.3-Flash", ["spark", "spark2"])
    ]
    assert "spans advertised by spark" == doc["split_brains"][0]["evidence"]
    by = {m["name"]: m for m in doc["members"]}
    # remote memory read from each member's own /health
    assert by["spark"]["memory"] == {
        "total_bytes": 128 * 2**30,
        "available_bytes": 40 * 2**30,
        "used_bytes": 88 * 2**30,
    }
    assert by["spark2"]["memory"] is None  # inconsistent figures -> unknown
    assert by["thor"]["memory"] is None  # not advertised -> unknown, never a guess
    assert by["core"]["memory"] is None  # local /proc fallback read nothing
    assert doc["hub"] == {
        "member": "core",
        "evidence": "declared: lobes.culture.dev",
        "declared": True,
    }
    # the evidence rides the three GETs per member the site already made
    assert len(opener.calls) == 3 * 4


def test_remote_declared_tunnel_makes_that_member_the_hub(tmp_path, monkeypatch):
    routes = _fleet_routes()
    routes["http://127.0.0.1:8001"]["/health"].pop("tunnel")
    routes["http://thor:8000"]["/health"]["tunnel"] = {
        "public_url": "https://thor.example.org",
        "declared": True,
    }
    src, _ = _fleet_sources(tmp_path, monkeypatch, routes)
    hub = site.build_mesh(src)["hub"]
    assert hub == {"member": "thor", "evidence": "declared: thor.example.org", "declared": True}


def test_local_falls_back_to_proc_and_tunnel_probe(tmp_path, monkeypatch):
    routes = _fleet_routes()
    routes["http://127.0.0.1:8001"]["/health"].pop("tunnel")
    mem = {"total_bytes": 64 * 2**30, "available_bytes": 16 * 2**30}
    src, _ = _fleet_sources(
        tmp_path,
        monkeypatch,
        routes,
        local_tunnel="container cloudflared running",
        local_mem=mem,
    )
    doc = site.build_mesh(src)
    assert {m["name"]: m for m in doc["members"]}["core"]["memory"]["total_bytes"] == 64 * 2**30
    assert doc["hub"] == {"member": "core", "evidence": "container cloudflared running"}


def test_remote_member_never_uses_the_local_probes(tmp_path, monkeypatch):
    routes = _fleet_routes()
    routes["http://127.0.0.1:8001"]["/health"].pop("tunnel")
    src, _ = _fleet_sources(
        tmp_path,
        monkeypatch,
        routes,
        local_tunnel="container cloudflared running",
        local_mem={"total_bytes": 64, "available_bytes": 16},
    )
    by = {m["name"]: m for m in site.build_mesh(src)["members"]}
    assert by["thor"]["memory"] is None
