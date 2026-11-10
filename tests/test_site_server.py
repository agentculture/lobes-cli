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
from lobes.site import _topology
from lobes.site import server as site

SECRET = "fake-mesh-key-for-tests-0000"
GW_SECRET = "fake-gateway-key-for-tests-0000"

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
            "cortex": _role("zai/GLM-5.3-Flash", proxied=True, hosted_by="http://spark:8001"),
            "tts": _role("ResembleAI/chatterbox", proxied=True),
        },
        # live 2026-11-10: spark's gateway advertises the TensorFold cortex;
        # spark2 (rank 0 of the same engine) does not list cortex at all
        "http://spark:8001": {"cortex": _role("zai/GLM-5.3-Flash", runtime="tensorfold")},
        "http://spark2:8000": {
            "stt-en": _role("nvidia/parakeet-tdt-0.6b-v2", runtime="parakeet"),
        },
        "http://thor:8000": {
            "reranker": _role("Qwen/Qwen3-Reranker-0.6B"),
            "worker": _role("Nemotron-Super"),
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


def _tf(rank, world=2, model="GLM-5.3-Flash", group="TF_GLM"):
    return _topology.Container(
        name="glm53-spark",
        image="tensorfold-glm53-spark:2026-11-01",
        model=model,
        ranks=((group, world, rank),),
    )


def _vllm_tp2():
    return _topology.Container(
        name="vllm-worker", image="vllm/vllm-openai:v0.31.0", model="Nemotron-Super", tp=2
    )


def _cloudflared(name="lobes-cloudflared"):
    return _topology.Container(name=name, image="cloudflare/cloudflared:2026.10.0", model=None)


def _topo(**over):
    topo = {
        "cultureos": [],
        "spark": [_tf(1)],
        "spark2": [_tf(0)],
        "thor": [_vllm_tp2()],
        "orin": None,  # unknown: never guessed
    }
    topo.update(over)
    return topo


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
        topology=lambda: _topo(),
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
    # spark2 does not advertise cortex; it holds rank 0 of spark's engine, so the
    # split gives it a cortex lobe the graph can draw
    assert spark2 == {"cortex": "zai/GLM-5.3-Flash", "stt-en": "nvidia/parakeet-tdt-0.6b-v2"}


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


def test_split_brain_from_tensorfold_world_rank(serve):
    _, data = _json(serve(), "/api/mesh")
    assert data["split_brains"] == [
        {
            "model": "zai/GLM-5.3-Flash",
            "served_name": "GLM-5.3-Flash",
            "roles": ["cortex"],
            "members": ["spark2", "spark"],  # rank order
            "evidence": "TensorFold TF_GLM_WORLD=2: spark2 rank 0, spark rank 1",
        }
    ]
    by = {m["name"]: m for m in data["members"]}
    assert {lobe["shape"] for lobe in by["spark"]["lobes"]} == {"split"}
    spark2 = {lobe["role"]: lobe for lobe in by["spark2"]["lobes"]}
    assert spark2["cortex"]["shape"] == "split"
    assert spark2["cortex"]["advertised"] is False  # drawn from topology, not its gateway
    assert spark2["stt-en"]["shape"] == "single"


def test_member_topology_is_the_allowlisted_projection(serve):
    _, data = _json(serve(), "/api/mesh")
    by = {m["name"]: m for m in data["members"]}
    assert by["spark2"]["topology"] == [
        {
            "name": "glm53-spark",
            "image": "tensorfold-glm53-spark:2026-11-01",
            "model": "GLM-5.3-Flash",
            "world_rank": [{"group": "TF_GLM", "world": 2, "rank": 0}],
            "tp": None,
            "pp": None,
        }
    ]
    assert by["orin"]["topology"] is None  # unknown
    assert by["cultureos"]["topology"] == []


def test_pair_is_one_model_pooled_on_several(serve):
    _, data = _json(serve(), "/api/mesh")
    assert [(p["model"], p["members"]) for p in data["pairs"]] == [
        ("Qwen/Qwen3-Reranker-0.6B", ["thor", "orin"])
    ]
    # the split model is not also reported as a pair
    assert all("GLM" not in p["model"] for p in data["pairs"])
    by = {m["name"]: m for m in data["members"]}
    shapes = {lobe["role"]: lobe["shape"] for lobe in by["orin"]["lobes"]}
    assert shapes == {"reranker": "pair", "embedder": "single"}


def _spark2_hosts_cortex():
    caps = _caps()
    caps["http://spark2:8000"]["cortex"] = _role("zai/GLM-5.3-Flash")
    return caps


def test_no_topology_evidence_means_pair_not_split(serve, tmp_path):
    src = make_sources(tmp_path, observe=make_observe(caps=_spark2_hosts_cortex()), topology=dict)
    _, data = _json(serve(src), "/api/mesh")
    assert data["split_brains"] == []
    assert ("zai/GLM-5.3-Flash", ["spark", "spark2"]) in [
        (p["model"], p["members"]) for p in data["pairs"]
    ]


@pytest.mark.parametrize(
    "spark, spark2, why",
    [
        ([_tf(1)], [_tf(0, world=3)], "mismatched world"),
        ([_tf(0)], [_tf(0)], "duplicate rank"),
        ([_tf(1)], [_tf(2)], "ranks do not cover 0..N-1"),
        ([_tf(1)], [], "world 2 on one member alone"),
        ([_tf(1)], None, "the other member is unknown"),
        ([_tf(1)], [_tf(0, model="Other-Model")], "different served model"),
        ([_tf(1)], [_tf(0, group="TF_QWEN")], "different engine group"),
    ],
)
def test_no_split_without_a_complete_matching_world(serve, tmp_path, spark, spark2, why):
    src = make_sources(tmp_path, topology=lambda: _topo(spark=spark, spark2=spark2))
    _, data = _json(serve(src), "/api/mesh")
    assert data["split_brains"] == [], why
    by = {m["name"]: m for m in data["members"]}
    assert by["spark"]["lobes"][0]["shape"] == "single", why
    assert "cortex" not in {lobe["role"] for lobe in by["spark2"]["lobes"]}, why


def test_one_box_tensor_parallel_is_not_a_split(serve, tmp_path):
    # thor's vLLM runs TP=2 on its own GPUs: recorded, never a cross-box split
    _, data = _json(serve(), "/api/mesh")
    assert all("Nemotron" not in s["model"] for s in data["split_brains"])
    by = {m["name"]: m for m in data["members"]}
    assert by["thor"]["topology"][0]["tp"] == 2
    assert {lobe["role"]: lobe["shape"] for lobe in by["thor"]["lobes"]}["worker"] == "single"


def test_split_with_no_advertising_member_has_no_roles(serve, tmp_path):
    caps = _caps()
    caps["http://spark:8001"].pop("cortex")
    src = make_sources(tmp_path, observe=make_observe(caps=caps))
    _, data = _json(serve(src), "/api/mesh")
    assert data["split_brains"] == [
        {
            "model": "GLM-5.3-Flash",
            "served_name": "GLM-5.3-Flash",
            "roles": [],
            "members": ["spark2", "spark"],
            "evidence": "TensorFold TF_GLM_WORLD=2: spark2 rank 0, spark rank 1",
        }
    ]
    by = {m["name"]: m for m in data["members"]}
    assert by["spark"]["lobes"] == []  # no role to draw it under


@pytest.mark.parametrize(
    "served, advertised, match",
    [
        ("GLM-5.3-Flash", "zai/GLM-5.3-Flash", True),
        ("glm_5.3_flash", "zai/GLM-5.3-Flash", True),
        ("/model", "zai/GLM-5.3-Flash", False),
        ("GLM-5.3", "zai/GLM-5.3-Flash", False),
        ("zai/GLM-5.3-Flash", "other/GLM-5.3-Flash", True),
    ],
)
def test_served_name_matches_on_the_final_path_segment(served, advertised, match):
    assert site._same_model(served, advertised) is match


def test_hub_only_from_a_running_lobes_cloudflared(serve, tmp_path):
    _, data = _json(serve(), "/api/mesh")
    assert data["hub"] is None
    src = make_sources(tmp_path, topology=lambda: _topo(cultureos=[_cloudflared()]))
    _, data = _json(serve(src), "/api/mesh")
    assert data["hub"] == {"member": "cultureos", "evidence": "container lobes-cloudflared running"}
    # a remote member running it is the hub just the same
    src = make_sources(tmp_path, topology=lambda: _topo(thor=[_vllm_tp2(), _cloudflared()]))
    _, data = _json(serve(src), "/api/mesh")
    assert data["hub"]["member"] == "thor"


def test_other_projects_cloudflared_is_not_the_hub(serve, tmp_path):
    foreign = [
        _cloudflared("xteink-cloudflared-ui-1"),
        _cloudflared("culture-library-cloudflared-1"),
    ]
    src = make_sources(tmp_path, topology=lambda: _topo(cultureos=foreign))
    _, data = _json(serve(src), "/api/mesh")
    assert data["hub"] is None


def test_two_tunnel_holders_is_no_hub(serve, tmp_path):
    src = make_sources(
        tmp_path, topology=lambda: _topo(cultureos=[_cloudflared()], spark2=[_cloudflared()])
    )
    _, data = _json(serve(src), "/api/mesh")
    assert data["hub"] is None


def test_topology_of_a_non_member_is_ignored(serve, tmp_path):
    src = make_sources(tmp_path, topology=lambda: _topo(stranger=[_cloudflared()]))
    _, data = _json(serve(src), "/api/mesh")
    assert data["hub"] is None


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

    leaky = _topology.Container(name="c", image=f"img:{SECRET}", model=f"m/{GW_SECRET}")
    src = make_sources(
        tmp_path,
        observe=make_observe(caps=caps),
        journal=lambda: j,
        export_listing=listing,
        topology=lambda: _topo(thor=[leaky]),
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


# ------------------------------- gateway evidence (t21) and topology (d2)


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
    """spark's /capabilities as the real gateway renders it (no declared keys)."""
    from lobes.gateway import server as gw
    from lobes.gateway._config import build_config
    from lobes.gateway._mesh_routing import RoutingSnapshot

    env = {
        "PRIMARY_BASE_URL": "http://10.77.77.2:8010",
        "PRIMARY_SERVED_NAME": "zai/GLM-5.3-Flash",
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
                {}, version="0.95.0", meminfo=Path("/nonexistent/meminfo")
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


def _tf_row(rank):
    return {
        "name": "glm53-spark",
        "image": "tensorfold-glm53-spark:2026-11-01",
        "model": "GLM-5.3-Flash",
        "env": {"TF_GLM_WORLD": "2", "TF_GLM_RANK": str(rank)},
    }


CF_ROW = {
    "name": "lobes-cloudflared",
    "image": "cloudflare/cloudflared:1",
    "model": None,
    "env": {},
}


def _topology_runner(by_dest):
    """Fake transport runner: ``local`` or the ssh destination -> projection rows."""
    calls = []

    def runner(argv, timeout):
        dest = "local" if argv[0] == "sh" else argv[5]
        calls.append(dest)
        rows = by_dest.get(dest)
        if rows is None:
            return 255, "", "ssh: unreachable"
        return 0, json.dumps(rows), ""

    runner.calls = calls
    return runner


def _fleet_sources(tmp_path, monkeypatch, routes, *, topology=None, local_mem=None):
    (tmp_path / ".env").write_text("VLLM_PORT=8001\nLOBES_MESH_NAME=core\n")
    site._residents.save(
        Residents(
            residents={
                "spark": _entry("spark", "spark", 8001, "/home/s/.local/bin/lobes", True),
                "spark2": _entry("spark2", None, 8000, "/x/lobes", True),
                "thor": _entry("thor", "thor", 8000, "/x/lobes", True),
            }
        ),
        tmp_path,
    )
    monkeypatch.setattr(site, "_read_meminfo", lambda path=None: local_mem)
    runner = _topology_runner(
        topology
        if topology is not None
        else {"local": [CF_ROW], "spark@spark": [_tf_row(1)], "spark2": [_tf_row(0)]}
    )
    opener = _GatewayOpener(routes)
    src = site.default_sources(tmp_path, opener=opener, topology_runner=runner)
    return src, opener, runner


def test_gateway_shape_end_to_end_split_memory_and_hub(tmp_path, monkeypatch):
    src, opener, runner = _fleet_sources(tmp_path, monkeypatch, _fleet_routes())
    startup = len(opener.calls)  # one local read at startup resolves this box's name
    assert startup == 3
    doc = site.build_mesh(src)
    assert doc["split_brains"] == [
        {
            "model": "zai/GLM-5.3-Flash",
            "served_name": "GLM-5.3-Flash",
            "roles": ["cortex"],
            "members": ["spark2", "spark"],
            "evidence": "TensorFold TF_GLM_WORLD=2: spark2 rank 0, spark rank 1",
        }
    ]
    by = {m["name"]: m for m in doc["members"]}
    assert [lobe["shape"] for lobe in by["spark2"]["lobes"] if lobe["role"] == "cortex"] == [
        "split"
    ]
    # remote memory read from each member's own /health
    assert by["spark"]["memory"] == {
        "total_bytes": 128 * 2**30,
        "available_bytes": 40 * 2**30,
        "used_bytes": 88 * 2**30,
    }
    assert by["spark2"]["memory"] is None  # inconsistent figures -> unknown
    assert by["thor"]["memory"] is None  # not advertised -> unknown, never a guess
    assert by["core"]["memory"] is None  # local /proc fallback read nothing
    assert by["thor"]["topology"] is None  # ssh failed -> unknown
    assert doc["hub"] == {"member": "core", "evidence": "container lobes-cloudflared running"}
    # the gateway evidence rides the three GETs per member the site already made
    assert len(opener.calls) - startup == 3 * 4
    # one topology command per member: residents over ssh, this box locally
    assert sorted(runner.calls) == ["local", "spark2", "spark@spark", "thor@thor"]


def test_topology_is_cached_between_page_loads(tmp_path, monkeypatch):
    src, _, runner = _fleet_sources(tmp_path, monkeypatch, _fleet_routes())
    site.build_mesh(src)
    site.build_mesh(src)
    assert len(runner.calls) == 4


def test_a_retired_tunnel_declaration_on_health_is_ignored(tmp_path, monkeypatch):
    routes = _fleet_routes()
    routes["http://thor:8000"]["/health"]["tunnel"] = {
        "public_url": "https://thor.example.org",
        "declared": True,
    }
    src, _, _ = _fleet_sources(
        tmp_path, monkeypatch, routes, topology={"spark@spark": [], "spark2": []}
    )
    assert site.build_mesh(src)["hub"] is None


def test_a_remote_lobes_cloudflared_makes_that_member_the_hub(tmp_path, monkeypatch):
    src, _, _ = _fleet_sources(
        tmp_path, monkeypatch, _fleet_routes(), topology={"local": [], "thor@thor": [CF_ROW]}
    )
    assert site.build_mesh(src)["hub"] == {
        "member": "thor",
        "evidence": "container lobes-cloudflared running",
    }


def test_local_memory_falls_back_to_proc(tmp_path, monkeypatch):
    mem = {"total_bytes": 64 * 2**30, "available_bytes": 16 * 2**30}
    src, _, _ = _fleet_sources(tmp_path, monkeypatch, _fleet_routes(), local_mem=mem)
    doc = site.build_mesh(src)
    assert {m["name"]: m for m in doc["members"]}["core"]["memory"]["total_bytes"] == 64 * 2**30


def test_remote_member_never_uses_the_local_probes(tmp_path, monkeypatch):
    src, _, _ = _fleet_sources(
        tmp_path,
        monkeypatch,
        _fleet_routes(),
        local_mem={"total_bytes": 64, "available_bytes": 16},
    )
    by = {m["name"]: m for m in site.build_mesh(src)["members"]}
    assert by["thor"]["memory"] is None


def test_topology_targets_from_residents_and_the_local_box(tmp_path):
    site._residents.save(
        Residents(
            residents={
                "spark": _entry("spark", "spark", 8001, "/x/lobes", True),
                "orin": _entry(None, None, 8000, None, True),  # ssh needs_you
                "cultureos": _entry("local", "a", 8001, "/x/lobes", True),  # not this box
            },
            guests={"guest": _entry("g", None, 8000, None, True)},
        ),
        tmp_path,
    )
    targets = site._topology_targets(tmp_path, "core")
    assert targets["spark"].ssh_target == "spark" and targets["spark"].user == "spark"
    assert targets["orin"] is None
    assert targets["core"].local is True
    assert "cultureos" not in targets  # ssh=local but the running gateway says core
    assert "guest" not in targets  # residents only


def test_stale_env_name_that_is_a_roster_peer_is_not_this_box(tmp_path, monkeypatch, capsys):
    # live 2026-11-10: core's .env said LOBES_MESH_NAME=spark; the gateway runs as core
    (tmp_path / ".env").write_text("VLLM_PORT=8001\nLOBES_MESH_NAME=spark\n")
    monkeypatch.setattr(site, "_read_meminfo", lambda path=None: None)
    opener = _GatewayOpener(_fleet_routes())
    src = site.default_sources(
        tmp_path, opener=opener, inspect=lambda: "core", topology_runner=_topology_runner({})
    )
    doc = site.build_mesh(src)
    assert doc["self"] == "core"
    by = {m["name"]: m for m in doc["members"]}
    assert by["spark"]["local"] is False
    assert by["spark"]["origin"] == "http://spark:8001"
    assert "disagrees with the running gateway" in capsys.readouterr().err


def test_peer_name_with_no_container_leaves_this_box_unnamed(tmp_path, monkeypatch, capsys):
    (tmp_path / ".env").write_text("VLLM_PORT=8001\nLOBES_MESH_NAME=spark\n")
    monkeypatch.setattr(site, "_read_meminfo", lambda path=None: None)
    opener = _GatewayOpener(_fleet_routes())
    src = site.default_sources(tmp_path, opener=opener, inspect=lambda: None)
    assert src.local_name is None
    assert "is a peer" in capsys.readouterr().err


def test_lanes_advertised_but_not_ready_are_not_pairs():
    members = [
        {"name": n, "lobes": [{"role": "stt", "model": "parakeet", "ready": r, "shape": "single"}]}
        for n, r in (("a", False), ("b", False), ("c", True))
    ]
    splits, pairs = site._shapes(members)
    assert pairs == [] and splits == []
    members[1]["lobes"][0]["ready"] = True
    _, pairs = site._shapes(members)
    assert pairs == [{"model": "parakeet", "roles": ["stt"], "members": ["b", "c"]}]


def test_gateway_only_member_is_a_gateway_not_a_brain():
    # live 2026-11-10, core: every role proxied; stt/tts listed with no backend wired
    caps = {
        "cortex": {"model": "zai/GLM-5.3-Flash", "feasible": False, "proxied": True},
        "worker": {"model": "w", "feasible": False, "hosted_by": "http://thor:8000"},
        "stt": {
            "model": "nvidia/parakeet-tdt-0.6b-v2",
            "feasible": True,
            "ready": False,
            "loaded": False,
            "endpoint": "",
        },
    }
    obs = MemberObservation(
        origin="http://127.0.0.1:8001",
        reachable=True,
        version="0.94.0",
        capabilities=caps,
        roster=[],
        health={},
    )
    src = site.Sources(
        local_origin="http://127.0.0.1:8001",
        local_name="core",
        observe=None,
        residents=None,
        journal=None,
        export_listing=None,
    )
    m = site._member(
        src,
        {"name": "core", "origin": "http://127.0.0.1:8001"},
        obs,
        True,
        site._residents.Residents(),
    )
    assert m["lobes"] == [] and m["kind"] == "gateway"
    assert m["routes"] == ["cortex", "worker"]


def test_a_wired_lane_that_is_down_is_still_a_lobe():
    caps = {
        "stt": {
            "model": "p",
            "feasible": True,
            "ready": False,
            "loaded": True,
            "endpoint": "http://orin:8000",
        }
    }
    obs = MemberObservation(
        origin="http://orin:8000",
        reachable=True,
        version="0.94.0",
        capabilities=caps,
        roster=[],
        health={},
    )
    src = site.Sources(
        local_origin="http://127.0.0.1:8001",
        local_name="core",
        observe=None,
        residents=None,
        journal=None,
        export_listing=None,
    )
    m = site._member(
        src, {"name": "orin", "origin": "http://orin:8000"}, obs, False, site._residents.Residents()
    )
    assert [lobe["role"] for lobe in m["lobes"]] == ["stt"] and m["kind"] == "brain"
