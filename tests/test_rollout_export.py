"""Lobe export (rollout t16): role -> compose service -> Dockerfile + inputs."""

from __future__ import annotations

import json
import shlex
from pathlib import Path

from lobes.cli import main
from lobes.rollout import _export
from lobes.rollout._transport import Target

SECRET = "sk-fake-secret-value-123456"
DEPLOY = "/home/u/.lobes"
LOBES = "/home/u/.local/bin/lobes"


def _config(ctx=DEPLOY):
    """What `docker compose config --no-interpolate --format json` returns."""
    return {
        "services": {
            "gateway": {
                "image": "lobes-gateway",
                "container_name": "model-gear-gateway",
                "environment": {"GATEWAY_API_KEY": "${GATEWAY_API_KEY:-}"},
            },
            "vllm-rerank": {
                "image": "vllm/vllm-openai@sha256:abc",
                "volumes": [
                    {"type": "bind", "source": f"{DEPLOY}/qwen3_reranker.jinja", "target": "/a"},
                    {"type": "bind", "source": f"{DEPLOY}/logs", "target": "/logs"},
                ],
                "command": ["--model", "x"],
            },
            "stt": {
                "build": {"context": ctx, "dockerfile": "Dockerfile.parakeet"},
                "container_name": "model-gear-stt",
                "environment": {
                    "HF_TOKEN": SECRET,
                    "PARAKEET_PORT": "${PARAKEET_PORT:-9002}",
                    "OPENAI_API_KEY": "${OPENAI_API_KEY:-EMPTY}",
                },
            },
        }
    }


DOCKERFILE = """\
FROM nvcr.io/nvidia/nemo:25
COPY listen_server.py /app/listen_server.py
COPY --chown=1:1 _readiness.py /app/_readiness.py
COPY --from=builder /x /y
CMD ["python3", "/app/listen_server.py"]
"""
INSPECT = [
    {
        "Name": "/super-worker",
        "Config": {
            "Image": "vllm/vllm-openai:v0.32.0-ubuntu2404",
            "Env": ["HF_TOKEN=%s" % SECRET, "FOO=bar", "VLLM_API_KEY=" + SECRET],
            "Cmd": ["--model", "m", "--port", "8000"],
            "Entrypoint": None,
        },
        "HostConfig": {
            "RestartPolicy": {"Name": "unless-stopped"},
            "Binds": ["/data/hf:/root/.cache/huggingface"],
            "PortBindings": {"8000/tcp": [{"HostIp": "", "HostPort": "8001"}]},
            "IpcMode": "host",
            "NetworkMode": "default",
            "DeviceRequests": [{"Count": -1, "Capabilities": [["gpu"]]}],
        },
    }
]


def _files(extra=None):
    f = {
        f"{DEPLOY}/Dockerfile.parakeet": DOCKERFILE,
        f"{DEPLOY}/listen_server.py": f"# server\nTOKEN = '{SECRET}'\n",
        f"{DEPLOY}/_readiness.py": "# ready\n",
        f"{DEPLOY}/qwen3_reranker.jinja": "{{ prompt }}\n",
        f"{DEPLOY}/.env": f"HF_TOKEN={SECRET}\n",
    }
    f.update(extra or {})
    return f


def make_runner(files=None, containers=None, with_gateway=True, config=None):
    files = _files() if files is None else files
    containers = INSPECT if containers is None else containers
    calls = []

    def runner(argv, timeout):
        argv = list(argv)
        if argv[0] == "ssh":
            argv = shlex.split(argv[-1])
        calls.append(argv)
        if argv[:1] == [LOBES]:
            if argv[1:4] == ["fleet", "files", "--json"]:
                chain = ["-f", "docker-compose.yml", "-f", "docker-compose.audio.yml"]
                return 0, json.dumps({"deployment_dir": DEPLOY, "files": chain}), ""
            return 2, "", "unknown"
        if argv[:2] == ["docker", "compose"]:
            return 0, json.dumps(config or _config()), ""
        if argv[0] == "cat":
            if argv[1] in files:
                return 0, files[argv[1]], ""
            return 1, "", "No such file"
        if argv[:2] == ["docker", "inspect"]:
            name = argv[-1]
            if name == "model-gear-gateway" and with_gateway:
                env = [
                    "STT_BASE_URL=http://stt:9002",
                    "WORKER_BASE_URL=http://super-worker:8000",
                    "GATEWAY_API_KEY=" + SECRET,
                ]
                return 0, json.dumps([{"Config": {"Env": env}}]), ""
            if name == "super-worker":
                return 0, json.dumps(containers), ""
            return 1, "", "no such container"
        return 1, "", "unexpected"

    runner.calls = calls
    return runner


TARGET = Target(ssh_target="spark2", user="u", lobes_path=LOBES)


def test_built_lobe_parakeet_exports_dockerfile_inputs_and_service():
    r = _export.export_role(
        "stt-en", TARGET, member="spark2", runner=make_runner(), secrets=[SECRET]
    )
    assert r.status == "built"
    assert r.service == "stt"
    names = set(r.files)
    assert {
        "Dockerfile.parakeet",
        "listen_server.py",
        "_readiness.py",
        "compose-service.json",
    } <= names
    assert "manifest.json" in names
    svc = r.files["compose-service.json"]
    assert "Dockerfile.parakeet" in svc
    # ${VAR} references stay as they are
    assert "${PARAKEET_PORT:-9002}" in svc
    assert "${OPENAI_API_KEY:-EMPTY}" in svc
    assert ".env" not in r.files and ".secrets.env" not in r.files


def test_no_secret_value_anywhere():
    r = _export.export_role(
        "stt-en", TARGET, member="spark2", runner=make_runner(), secrets=[SECRET]
    )
    for name, text in r.files.items():
        assert SECRET not in text, name
    w = _export.export_role("worker", TARGET, member="thor", runner=make_runner(), secrets=[SECRET])
    for name, text in w.files.items():
        assert SECRET not in text, name


def test_image_only_container_outside_compose():
    r = _export.export_role("worker", TARGET, member="thor", runner=make_runner(), secrets=[SECRET])
    assert r.status == "image-only"
    assert r.source == "container"
    assert r.image == "vllm/vllm-openai:v0.32.0-ubuntu2404"
    run = r.files["docker-run.sh"]
    assert run.startswith("#!/bin/sh")
    assert "docker run" in run and "--name super-worker" in run
    assert "vllm/vllm-openai:v0.32.0-ubuntu2404" in run
    assert "-p 8001:8000" in run and "--ipc host" in run and "--gpus all" in run
    assert "FOO=bar" in run
    assert "HF_TOKEN=${HF_TOKEN}" in run
    man = json.loads(r.files["manifest.json"])
    assert man["status"] == "image-only" and man["image"] == r.image


def test_image_only_compose_service_with_mounted_file():
    r = _export.export_role(
        "reranker", TARGET, member="orin", runner=make_runner(), secrets=[SECRET]
    )
    assert r.status == "image-only"
    assert r.source == "compose"
    assert r.service == "vllm-rerank"
    assert r.image == "vllm/vllm-openai@sha256:abc"
    assert "qwen3_reranker.jinja" in r.files
    assert "compose-service.json" in r.files
    # logs dir is not a file: reported, not fatal
    assert any("logs" in n for n in r.notes)


def test_unmapped_role_is_reported_not_guessed():
    r = _export.export_role("muse", TARGET, member="spark2", runner=make_runner(), secrets=[SECRET])
    assert r.status == "unmapped"
    assert set(r.files) == {"manifest.json"} and r.reason


def test_convention_fallback_without_gateway():
    r = _export.export_role(
        "stt-en", TARGET, member="spark2", runner=make_runner(with_gateway=False), secrets=[SECRET]
    )
    assert r.status == "built" and r.service == "stt"
    assert any("convention" in n for n in r.notes)


def test_export_member_bundle_and_write(tmp_path: Path):
    b = _export.export_member(
        ["stt-en", "worker", "muse"],
        TARGET,
        member="spark2",
        runner=make_runner(),
        secrets=[SECRET],
    )
    assert [x.status for x in b.exports] == ["built", "image-only", "unmapped"]
    paths = b.paths()
    assert "spark2/stt-en/Dockerfile.parakeet" in paths
    assert "spark2/worker/docker-run.sh" in paths
    assert "spark2/manifest.json" in paths
    written = b.write(tmp_path)
    assert (tmp_path / "spark2/stt-en/listen_server.py").is_file()
    assert sorted(written) == sorted(paths)
    for p in tmp_path.rglob("*"):
        if p.is_file():
            assert SECRET not in p.read_text()


def test_build_path_escape_refused():
    r = _export.export_role(
        "stt-en", TARGET, member="m", runner=make_runner(config=_config("/etc")), secrets=[]
    )
    assert "Dockerfile.parakeet" not in r.files
    assert any("escape" in n for n in r.notes)


def test_compose_config_uses_no_interpolate_and_no_env_files():
    runner = make_runner()
    _export.export_role("stt-en", TARGET, member="m", runner=runner, secrets=[])
    cfg = [c for c in runner.calls if c[:2] == ["docker", "compose"]]
    assert cfg and "--no-interpolate" in cfg[0] and "json" in cfg[0]
    assert f"{DEPLOY}/docker-compose.audio.yml" in cfg[0]
    assert not any(c[0] == "cat" and c[1].endswith(".env") for c in runner.calls)


def test_dockerfile_copy_parsing():
    srcs = _export.dockerfile_copy_sources(
        DOCKERFILE + "COPY a.py b.py /x/\nCOPY . .\nCOPY *.py /z\n"
    )
    assert srcs.files == ["listen_server.py", "_readiness.py", "a.py", "b.py"]
    assert srcs.skipped


def test_redact_text_keeps_refs_and_scrubs_literals():
    t = "- OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}\n- HF_TOKEN=hf_literal\nX=%s\n" % SECRET
    out = _export.redact_text(t, [SECRET])
    assert "${OPENAI_API_KEY:-EMPTY}" in out
    assert "hf_literal" not in out and SECRET not in out


def _patch_resident(tmp_path, monkeypatch, runner):
    (tmp_path / "residents.toml").write_text(
        '[resident.spark2]\nssh = "spark2"\nuser = "u"\nlobes_path = "%s"\n' % LOBES
    )
    monkeypatch.setattr(_export, "default_runner", runner)


def test_cli_dry_run_lists_files_and_apply_writes(tmp_path, monkeypatch, capsys):
    _patch_resident(tmp_path, monkeypatch, make_runner())
    out = tmp_path / "out"
    rc = main(
        [
            "lobe",
            "export",
            "stt-en",
            "--member",
            "spark2",
            "--base-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--json",
        ]
    )
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["dry_run"] is True
    assert "spark2/stt-en/Dockerfile.parakeet" in data["files"]
    assert not out.exists()
    rc = main(
        [
            "lobe",
            "export",
            "stt-en",
            "--member",
            "spark2",
            "--base-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--apply",
        ]
    )
    assert rc == 0
    assert (out / "spark2/stt-en/Dockerfile.parakeet").is_file()


def test_cli_unknown_member(tmp_path, monkeypatch, capsys):
    _patch_resident(tmp_path, monkeypatch, make_runner())
    rc = main(["lobe", "export", "stt-en", "--member", "nope", "--base-dir", str(tmp_path)])
    assert rc == 1
    assert "nope" in capsys.readouterr().err
