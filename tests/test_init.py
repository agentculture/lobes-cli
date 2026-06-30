"""Tests for ``lobes init`` — scaffold a deployment dir."""

from __future__ import annotations

import json
import stat

from lobes.cli import main
from lobes.runtime import _compose


def test_init_dry_run_writes_nothing(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", str(target)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out
    assert not target.exists()


def test_init_default_scaffolds_fleet_duo(tmp_path, capsys) -> None:
    # DEFAULT topology flip (issue #69): a bare `lobes init` now scaffolds the
    # fleet duo (main primary + multimodal gear + gateway + embed/rerank), NOT a
    # single model. The single-model path moved behind `--single` (see below).
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.yml").is_file()
    assert (target / ".env").is_file()
    # The whole FLEET_TEMPLATES set is written — including the gateway Dockerfile,
    # which the legacy single-model scaffold does NOT carry.
    written = {p.name for p in target.iterdir() if p.is_file()}
    assert set(_compose.FLEET_TEMPLATES.values()) <= written
    assert "Dockerfile.gateway" in written
    # The duo is present in the materialised compose: the Qwen primary + the
    # Gemma multimodal gear, fronted by the gateway.
    compose = (target / "docker-compose.yml").read_text()
    assert "vllm-primary" in compose
    assert "vllm-multimodal" in compose
    assert "model-gear-gateway" in compose
    # Durable logs (issue #50): wrapper scaffolded + each vLLM gear runs it.
    assert (target / "mg-logwrap.sh").is_file()
    assert (target / "logs").is_dir()
    assert 'entrypoint: ["bash", "/usr/local/bin/mg-logwrap"]' in compose


def test_init_default_apply_json_lists_fleet_files(tmp_path, capsys) -> None:
    # The default `init --apply --json` now lists the FLEET file set (the gateway
    # Dockerfile is the tell-tale fleet-only file) and reports fleet=True.
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scaffolded"] == str(target)
    assert payload["fleet"] is True
    assert payload["single"] is False
    assert set(payload["files"]) == set(_compose.FLEET_TEMPLATES.values())
    assert "Dockerfile.gateway" in payload["files"]


def test_init_single_scaffolds_legacy(tmp_path, capsys) -> None:
    # `lobes init --single` restores the legacy single-model scaffold (one vLLM
    # server, no gateway). This is the behaviour that used to be the default.
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--single", "--apply"])
    assert rc == 0
    written = {p.name for p in target.iterdir() if p.is_file()}
    assert set(_compose.SINGLE_TEMPLATES.values()) <= written
    # Legacy single-model has no gateway / fleet services.
    assert "Dockerfile.gateway" not in written
    compose = (target / "docker-compose.yml").read_text()
    assert "model-gear-vllm" in compose
    assert "vllm-primary" not in compose
    # OpenAI tool/function calling is enabled out of the box (issue #9); the
    # parser is env-driven so a switched model can override it (default qwen3_coder
    # for the Qwen3.6-27B primary).
    assert "--enable-auto-tool-choice" in compose
    assert "--tool-call-parser=${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" in compose
    assert "VLLM_TOOL_CALL_PARSER=qwen3_coder" in (target / ".env").read_text()
    # Durable logs (issue #50): the single vLLM service runs the wrapper + names
    # its own per-boot log file.
    assert 'entrypoint: ["bash", "/usr/local/bin/mg-logwrap"]' in compose
    assert "MG_LOG_NAME=vllm" in compose
    assert "/logs/model-gear" in compose


def test_init_legacy_alias_matches_single(tmp_path) -> None:
    # `--legacy` is an accepted alias for `--single`.
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--legacy", "--apply"])
    assert rc == 0
    written = {p.name for p in target.iterdir() if p.is_file()}
    assert set(_compose.SINGLE_TEMPLATES.values()) <= written
    assert "Dockerfile.gateway" not in written


def test_init_single_apply_json(tmp_path, capsys) -> None:
    target = tmp_path / "deploy"
    rc = main(["init", str(target), "--single", "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scaffolded"] == str(target)
    assert payload["fleet"] is False
    assert payload["single"] is True
    assert set(payload["files"]) == set(_compose.SINGLE_TEMPLATES.values())
    assert "Dockerfile.gateway" not in payload["files"]


def test_init_refuses_overwrite_without_force(tmp_path) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    rc = main(["init", str(target), "--apply"])
    assert rc == 1  # exists; needs --force


def test_init_force_overwrites(tmp_path) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    assert main(["init", str(target), "--apply", "--force"]) == 0


def test_init_default_target(capsys) -> None:
    # The autouse fixture points default_deployment_dir at an empty tmp dir.
    default = _compose.default_deployment_dir()
    rc = main(["init", "--apply"])
    assert rc == 0
    assert (default / "docker-compose.yml").is_file()
    assert (default / ".env").is_file()


def test_init_env_is_owner_only(tmp_path) -> None:
    target = tmp_path / "deploy"
    assert main(["init", str(target), "--apply"]) == 0
    mode = stat.S_IMODE((target / ".env").stat().st_mode)
    assert mode == 0o600  # .env may hold HF_TOKEN — not world-readable


def test_init_local_folder(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["init", ".", "--apply"])
    assert rc == 0
    assert (tmp_path / "docker-compose.yml").is_file()


# --- fleet scaffold -------------------------------------------------------


def test_init_fleet_apply_writes_three_files(tmp_path) -> None:
    from lobes import __version__

    target = tmp_path / "fleet"
    rc = main(["init", "--fleet", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.yml").is_file()
    assert (target / ".env").is_file()
    assert (target / "Dockerfile.gateway").is_file()
    compose = (target / "docker-compose.yml").read_text()
    assert "vllm-primary" in compose
    assert "model-gear-gateway" in compose
    # Single-backend by default: no fallback service is scaffolded (the compose
    # may mention vllm-fallback in "how to add one" comments, so check the
    # service's container_name, which only appears when the service is defined).
    assert "model-gear-vllm-fallback" not in compose
    # Durable logs (issue #50): wrapper scaffolded + each vLLM gear runs it + names
    # its own per-boot log file (primary/embed/rerank).
    assert (target / "mg-logwrap.sh").is_file()
    assert (target / "logs").is_dir()
    assert 'entrypoint: ["bash", "/usr/local/bin/mg-logwrap"]' in compose
    for svc in ("primary", "embed", "rerank"):
        assert f"MG_LOG_NAME={svc}" in compose
    env = (target / ".env").read_text()
    assert "PRIMARY_MODEL=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in env
    assert "FALLBACK_MODEL=" not in env
    # The primary is trimmed to 128K context and util 0.45 to make room for
    # the opt-in vllm-middle gear (see env.example GPU budget).
    assert "PRIMARY_GPU_MEM_UTIL=0.45" in env
    assert "PRIMARY_MAX_MODEL_LEN=131072" in env
    # init --fleet pins the gateway image to the running lobes-cli version.
    assert f"MODEL_GEAR_VERSION={__version__}" in env
    # coherence mirror keeps the single-model read-only verbs sensible.
    assert "VLLM_SERVED_NAME=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in env


def test_init_fleet_dry_run_json(tmp_path, capsys) -> None:
    target = tmp_path / "fleet"
    rc = main(["init", "--fleet", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fleet"] is True
    names = {f["name"] for f in payload["files"]}
    assert names == {
        "docker-compose.yml",
        ".env",
        "Dockerfile.gateway",
        "Dockerfile.vllm-gemma4",   # custom vLLM image for vllm-multimodal (issue #71)
        "mg-logwrap.sh",
        "cf-tunnel.env.example",
    }
    assert not target.exists()


# --- audio overlay (--fleet --audio) --------------------------------------


def test_init_audio_is_valid_by_default(tmp_path, capsys) -> None:
    # The fleet is now the default, and the audio overlay layers on the fleet, so
    # `lobes init --audio` (no --fleet needed) is now valid — it scaffolds the
    # fleet + audio overlay. (Was an error back when single was the default.)
    target = tmp_path / "fa"
    rc = main(["init", "--audio", str(target), "--apply"])
    assert rc == 0
    assert (target / "docker-compose.audio.yml").is_file()


def test_init_audio_incompatible_with_single(capsys) -> None:
    # The audio overlay needs the fleet; it cannot layer on the legacy single model.
    rc = main(
        ["init", "--single", "--audio", "/tmp/nope", "--json"]
    )  # nosec B108 - never written (errors first)
    assert rc == 1  # EXIT_USER_ERROR
    err = capsys.readouterr().err
    assert "--audio" in err and "--single" in err


def test_init_fleet_audio_apply_writes_overlay_and_appends_env(tmp_path) -> None:
    target = tmp_path / "fa"
    rc = main(["init", "--fleet", "--audio", str(target), "--apply"])
    assert rc == 0
    # fleet files + the audio overlay files. _readiness.py MUST be scaffolded:
    # Dockerfile.parakeet COPYs it, so a missing scaffold breaks `build stt`.
    for name in (
        "docker-compose.yml",
        "Dockerfile.gateway",
        "docker-compose.audio.yml",
        "Dockerfile.realtime",
        "Dockerfile.parakeet",
        "Dockerfile.chatterbox",
        "listen_server.py",
        "_readiness.py",
    ):
        assert (target / name).is_file(), name
    env = (target / ".env").read_text()
    # fleet keys still present, audio keys appended (not clobbered).
    assert "PRIMARY_MODEL=sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP" in env
    assert "CHATTERBOX_PORT=9000" in env
    assert "AUDIO_URL=http://realtime:8080" in env


def test_init_fleet_audio_dry_run_json_lists_overlay(tmp_path, capsys) -> None:
    target = tmp_path / "fa"
    rc = main(["init", "--fleet", "--audio", str(target), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["fleet"] is True and payload["audio"] is True
    names = {f["name"] for f in payload["files"]}
    # The dry-run JSON must list EVERY audio overlay file (derive from the source
    # map so a dropped/added template can't silently slip past this assertion).
    assert set(_compose.AUDIO_TEMPLATES.values()) <= names
    assert "Dockerfile.chatterbox" in names  # the file this PR wires in
    assert not target.exists()


def test_init_fleet_audio_dry_run_text_mentions_appended_env(tmp_path, capsys) -> None:
    target = tmp_path / "fa"
    rc = main(["init", "--fleet", "--audio", str(target)])  # text mode, no --apply
    assert rc == 0
    out = capsys.readouterr().out
    assert ".env (+ audio keys appended)" in out
    assert "Re-run with --apply to write." in out
    assert not target.exists()


def test_every_compose_referenced_dockerfile_is_scaffolded(tmp_path) -> None:
    """Root-cause guardrail for this PR: any `dockerfile:` a scaffolded compose
    references MUST itself be scaffolded, or `docker compose build` fails. Would
    have caught the omitted Dockerfile.chatterbox (and any future build file added
    to compose but forgotten in the template maps)."""
    import re

    templates = {**_compose.FLEET_TEMPLATES, **_compose.AUDIO_TEMPLATES}
    _compose.write_scaffold(tmp_path, force=True, templates=templates)
    for compose_name in (_compose.COMPOSE_FILE, _compose.AUDIO_OVERLAY):
        text = (tmp_path / compose_name).read_text(encoding="utf-8")
        for ref in re.findall(r"^\s*dockerfile:\s*(\S+)", text, re.MULTILINE):
            assert (tmp_path / ref).is_file(), f"{compose_name} builds from {ref}, not scaffolded"


# --- the default-on topology (issue #69) ----------------------------------


def test_fleet_compose_default_on_generate_services_are_the_duo() -> None:
    """The packaged fleet compose's DEFAULT-ON (no-``profiles``) generate services
    must be exactly the duo {vllm-primary, vllm-multimodal}; the legacy generate
    gears vllm-minor (4B) and vllm-middle (14B) must sit BEHIND profiles so a bare
    ``docker compose up -d`` (what `lobes serve` runs) does NOT start them."""
    from importlib.resources import files

    import yaml

    text = (files("lobes.templates") / "fleet" / "docker-compose.yml").read_text(encoding="utf-8")
    compose = yaml.safe_load(text)
    services = compose["services"]

    generate = {"vllm-primary", "vllm-multimodal", "vllm-minor", "vllm-middle"}
    # All four generate gears are defined in the compose.
    assert generate <= set(services)
    default_on_generate = {
        name for name in generate if name in services and not services[name].get("profiles")
    }
    assert default_on_generate == {"vllm-primary", "vllm-multimodal"}
    # The legacy generate gears are profile-gated (excluded from a bare up -d).
    assert services["vllm-minor"].get("profiles")
    assert services["vllm-middle"].get("profiles")
