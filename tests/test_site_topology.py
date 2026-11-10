"""lobes site topology reader (deviation d2): what each member's containers run.

The remote command is exercised for real against a fake ``docker`` on PATH, so
the allowlist filtering that happens ON THE MEMBER is tested, not mocked. The
reader itself is driven with injected runners and an injected clock.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import threading
from pathlib import Path

import pytest

from lobes.rollout._transport import Target
from lobes.site import _topology as T

# Secret-looking env values that must never leave the member.
HF = "hf_fakeTokenForTestsOnly0000"
GW = "fake-gateway-key-for-tests-0000"


def _container(name, image, cmd, env, entrypoint=None):
    return {
        "Id": name,
        "Name": "/" + name,
        "Config": {"Image": image, "Entrypoint": entrypoint, "Cmd": cmd, "Env": env},
    }


def glm(rank, world=2, name="glm53-spark", extra_env=()):
    return _container(
        name,
        "tensorfold-glm53-spark:2026-11-01",
        ["serve", "/model", "--name", "GLM-5.3-Flash", "--host", "0.0.0.0", "--port", "8010"],
        [
            "PATH=/usr/bin",
            f"TF_GLM_WORLD={world}",
            f"TF_GLM_RANK={rank}",
            f"HF_TOKEN={HF}",
            f"GATEWAY_API_KEY={GW}",
            *extra_env,
        ],
        entrypoint=["/usr/local/bin/tensorfold-native"],
    )


def vllm_tp2():
    return _container(
        "vllm-primary",
        "vllm/vllm-openai:v0.31.0",
        [
            "--model",
            "unsloth/Qwen3.8-27B-NVFP4",
            "--served-model-name",
            "unsloth/Qwen3.8-27B-NVFP4",
            "--tensor-parallel-size=2",
            "--api-key",
            GW,
        ],
        ["VLLM_API_KEY=" + GW],
    )


def cloudflared():
    return _container(
        "lobes-cloudflared",
        "cloudflare/cloudflared:2026.10.0",
        ["tunnel", "run"],
        ["TUNNEL_TOKEN=" + HF],
    )


# ------------------------------------------------- the real remote command


@pytest.fixture
def fake_docker(tmp_path):
    """A ``docker`` on PATH answering ``ps -q`` and ``inspect`` from a fixture."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    fixture = tmp_path / "inspect.json"
    script = bindir / "docker"
    script.write_text(
        "#!/bin/sh\n"
        'if [ -n "$FAKE_DOCKER_FAIL" ]; then echo "permission denied" >&2; exit 1; fi\n'
        'case "$1" in\n'
        "  ps) python3 -c 'import json,sys; "
        '[print(c["Id"]) for c in json.load(open(sys.argv[1]))]\''
        f' "{fixture}" ;;\n'
        f'  inspect) cat "{fixture}" ;;\n'
        "  *) exit 2 ;;\n"
        "esac\n"
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)

    def run(containers, fail=False):
        fixture.write_text(json.dumps(containers))
        env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}"}
        if fail:
            env["FAKE_DOCKER_FAIL"] = "1"
        argv = T.build_remote_argv()
        p = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=30)
        return p.returncode, p.stdout, p.stderr

    return run


def test_remote_command_projects_only_the_allowlist(fake_docker):
    rc, out, _ = fake_docker([glm(1), vllm_tp2(), cloudflared()])
    assert rc == 0
    # nothing but the allowlist leaves the member: no secret, no raw env, no Cmd
    for leaked in (HF, GW, "HF_TOKEN", "GATEWAY_API_KEY", "PATH=", "--api-key", "/usr/bin"):
        assert leaked not in out, leaked
    rows = json.loads(out)
    assert rows[0] == {
        "name": "glm53-spark",
        "image": "tensorfold-glm53-spark:2026-11-01",
        "model": "GLM-5.3-Flash",
        "env": {"TF_GLM_WORLD": "2", "TF_GLM_RANK": "1"},
    }
    assert rows[1]["model"] == "unsloth/Qwen3.8-27B-NVFP4"
    assert rows[1]["tp"] == 2
    assert rows[1]["env"] == {}
    assert rows[2] == {
        "name": "lobes-cloudflared",
        "image": "cloudflare/cloudflared:2026.10.0",
        "model": None,
        "env": {},
    }


def test_remote_command_drops_allowlist_keys_with_non_numeric_values(fake_docker):
    sneaky = glm(0, extra_env=(f"TF_GLM_SECRET={HF}", f"TF_X_WORLD={GW}", "TF_A_B_RANK=3"))
    rc, out, _ = fake_docker([sneaky])
    assert rc == 0
    assert HF not in out
    assert GW not in out
    assert json.loads(out)[0]["env"] == {
        "TF_GLM_WORLD": "2",
        "TF_GLM_RANK": "0",
        "TF_A_B_RANK": "3",
    }


def test_remote_command_with_no_containers_prints_an_empty_list(fake_docker):
    rc, out, _ = fake_docker([])
    assert rc == 0
    assert json.loads(out) == []


def test_remote_command_fails_when_docker_fails(fake_docker):
    rc, out, _ = fake_docker([glm(0)], fail=True)
    assert rc != 0
    assert out.strip() == ""


@pytest.mark.parametrize(
    "cmd, entry, model, tp, pp",
    [
        (["serve", "/model", "--name", "GLM-5.3-Flash"], None, "GLM-5.3-Flash", None, None),
        (["--served-model-name=a/b", "--model", "c/d"], None, "a/b", None, None),
        (["--model=c/d", "-tp", "4", "--pipeline-parallel-size", "2"], None, "c/d", 4, 2),
        (["sh", "-c", "vllm serve --model x/y --tensor-parallel-size 2"], None, "x/y", 2, None),
        (["serve", "x/positional"], ["vllm"], "x/positional", None, None),
        (["--name", "bad name; rm -rf"], None, None, None, None),
    ],
)
def test_remote_command_parses_served_name_and_parallelism(fake_docker, cmd, entry, model, tp, pp):
    rc, out, _ = fake_docker([_container("c", "img:1", cmd, [], entrypoint=entry)])
    assert rc == 0
    row = json.loads(out)[0]
    assert row["model"] == model
    assert row.get("tp") == tp
    assert row.get("pp") == pp


# ------------------------------------------------- local re-validation


def test_parse_revalidates_and_redacts_locally():
    remote = json.dumps(
        [
            {
                "name": "glm53-spark",
                "image": "img:1",
                "model": "GLM-5.3-Flash",
                "env": {"TF_GLM_WORLD": "2", "TF_GLM_RANK": "0", "HF_TOKEN": HF},
                "raw": GW,
            },
            {"name": "bad name", "image": "x", "model": None, "env": {}},
            {"name": "ok", "image": "img:" + GW, "model": None, "env": {"TF_Q_WORLD": GW}},
        ]
    )
    rows = T.parse_projection(remote, secrets=(GW,))
    dumped = json.dumps([r.as_dict() for r in rows])
    assert HF not in dumped
    assert GW not in dumped
    assert "HF_TOKEN" not in dumped
    assert [r.name for r in rows] == ["glm53-spark", "ok"]
    assert rows[0].ranks == (("TF_GLM", 2, 0),)
    assert rows[1].ranks == ()


@pytest.mark.parametrize("raw", ["", "not json", "{}", '[{"name": 3}]'])
def test_parse_garbage_is_unknown_or_skipped(raw):
    rows = T.parse_projection(raw)
    assert rows is None or rows == []


def test_rank_without_world_is_not_a_rank():
    rows = T.parse_projection(
        json.dumps([{"name": "a", "image": "i", "model": "m", "env": {"TF_GLM_RANK": "0"}}])
    )
    assert rows[0].ranks == ()


# ------------------------------------------------- the reader


def _projection(*containers):
    out = []
    for c in containers:
        row = {"name": c[0], "image": c[1], "model": c[2], "env": c[3]}
        out.append(row)
    return json.dumps(out)


SPARK = _projection(
    ("glm53-spark", "tf:1", "GLM-5.3-Flash", {"TF_GLM_WORLD": "2", "TF_GLM_RANK": "1"})
)
SPARK2 = _projection(
    ("glm53-spark", "tf:1", "GLM-5.3-Flash", {"TF_GLM_WORLD": "2", "TF_GLM_RANK": "0"})
)
CORE = _projection(("lobes-cloudflared", "cloudflare/cloudflared:1", None, {}))


class FakeRunner:
    def __init__(self, by_dest):
        self.by_dest = by_dest
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, argv, timeout):
        dest = "local" if argv[0] == "sh" else argv[5]
        with self.lock:
            self.calls.append((dest, timeout))
        answer = self.by_dest[dest]
        if isinstance(answer, Exception):
            raise answer
        return answer


def _targets():
    return {
        "core": Target(local=True),
        "spark": Target(ssh_target="spark", user="spark"),
        "spark2": Target(ssh_target="spark2"),
        "thor": Target(ssh_target="thor", user="thor"),
        "orin": None,  # ssh is needs_you in residents.toml
    }


def _runner():
    return FakeRunner(
        {
            "local": (0, CORE, ""),
            "spark@spark": (0, SPARK, ""),
            "spark2": (0, SPARK2, ""),
            "thor@thor": (255, "", "ssh: connect to host thor port 22: No route"),
        }
    )


def test_reader_reads_every_member_once_with_one_command_each():
    runner = _runner()
    reader = T.TopologyReader(_targets, runner=runner, timeout=7)
    topo = reader.read()
    assert [c.name for c in topo["core"]] == ["lobes-cloudflared"]
    assert topo["spark"][0].ranks == (("TF_GLM", 2, 1),)
    assert topo["spark2"][0].ranks == (("TF_GLM", 2, 0),)
    assert topo["thor"] is None  # ssh failure -> unknown, never guessed
    assert topo["orin"] is None  # no ssh target -> unknown
    assert sorted(d for d, _ in runner.calls) == ["local", "spark2", "spark@spark", "thor@thor"]
    assert all(t == 7 for _, t in runner.calls)


def test_reader_uses_batchmode_ssh_with_the_remote_command():
    seen = []

    def runner(argv, timeout):
        seen.append(list(argv))
        return 0, "[]", ""

    T.TopologyReader(lambda: {"spark": Target(ssh_target="spark")}, runner=runner).read()
    argv = seen[0]
    assert argv[:3] == ["ssh", "-o", "BatchMode=yes"]
    assert "docker ps -q" in argv[-1]
    assert "docker inspect" in argv[-1]


def test_reader_caches_for_the_ttl_then_refreshes():
    now = [1000.0]
    runner = _runner()
    reader = T.TopologyReader(_targets, runner=runner, ttl=120, clock=lambda: now[0])
    reader.read()
    first = len(runner.calls)
    now[0] += 119
    reader.read()
    assert len(runner.calls) == first  # within TTL: no ssh at all
    now[0] += 2
    reader.read()
    assert len(runner.calls) == 2 * first


def test_reader_failures_are_cached_too():
    now = [0.0]
    runner = _runner()
    reader = T.TopologyReader(_targets, runner=runner, clock=lambda: now[0])
    reader.read()
    reader.read()
    assert [d for d, _ in runner.calls].count("thor@thor") == 1


def test_reader_runner_exception_is_unknown():
    runner = _runner()
    runner.by_dest["spark@spark"] = OSError("ssh missing")
    topo = T.TopologyReader(_targets, runner=runner).read()
    assert topo["spark"] is None
    assert topo["spark2"] is not None


def test_reader_timeout_is_unknown():
    runner = _runner()
    runner.by_dest["spark2"] = (124, "", "timeout")
    assert T.TopologyReader(_targets, runner=runner).read()["spark2"] is None


def test_reader_redacts_runner_output_before_parsing():
    leak = _projection(("c", "img:" + GW, "m", {}))
    reader = T.TopologyReader(
        lambda: {"a": Target(local=True)},
        runner=lambda argv, timeout: (0, leak, ""),
        secrets=(GW,),
    )
    topo = reader.read()
    assert GW not in json.dumps([c.as_dict() for c in topo["a"]])


def test_reader_targets_failure_yields_empty_not_crash():
    def boom():
        raise OSError("residents unreadable")

    assert T.TopologyReader(boom, runner=_runner()).read() == {}


def test_as_dict_carries_no_env():
    row = T.parse_projection(SPARK)[0].as_dict()
    assert row == {
        "name": "glm53-spark",
        "image": "tf:1",
        "model": "GLM-5.3-Flash",
        "world_rank": [{"group": "TF_GLM", "world": 2, "rank": 1}],
        "tp": None,
        "pp": None,
    }


def test_module_is_stdlib_only():
    src = Path(T.__file__).read_text(encoding="utf-8")
    assert "import yaml" not in src
    assert "import requests" not in src
