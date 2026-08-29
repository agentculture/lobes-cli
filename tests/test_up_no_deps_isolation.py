"""A targeted operation touches ONLY what it names (issue #222).

Two independent defects let `lobes up <role>` — and a hand-written `docker
compose up -d gateway` — restart lobes they promise not to touch. Hit live on a
Jetson AGX Thor 2026-08-28: a gateway-only rebuild recreated the resident 27B
cortex (a multi-minute reload) and STARTED `vllm-multimodal` on a box whose own
`.env` says `MULTIMODAL_FEASIBLE=false` with an active proxy to a peer.

Both guards the issue asked for live here:

1. the role-targeted argv carries ``--no-deps``;
2. no service is a hard start-order dependency of the gateway, so a gateway
   (re)start can never drag an infeasible lane up with it.
"""

from __future__ import annotations

import json
from importlib.resources import files

import pytest
import yaml

from lobes.cli import main
from lobes.cli._commands import up as up_cmd
from lobes.runtime import _compose
from tests.test_cli_up import _ok, _scaffold_fleet, _scaffold_fleet_audio

# --- guard 1: the argv ------------------------------------------------------


@pytest.mark.parametrize("target", up_cmd.TARGETS)
def test_every_up_target_is_isolated_with_no_deps(target: str) -> None:
    services, _ = up_cmd._resolve(target)
    argv = _compose.compose_service_argv("up", [], services)
    assert _compose.NO_DEPS_FLAG in argv
    # ...and it precedes the service names, so it is parsed as a flag of `up`.
    assert argv.index(_compose.NO_DEPS_FLAG) < argv.index(services[0])


def test_stop_does_not_carry_no_deps() -> None:
    """`docker compose stop` never walks depends_on, so the flag would be noise
    — and `stop --no-deps` is not even a valid invocation."""
    argv = _compose.compose_service_argv("stop", [], ["vllm-primary"])
    assert argv == ["docker", "compose", "stop", "vllm-primary"]


def test_dry_run_plan_and_applied_argv_are_the_same_command(tmp_path, monkeypatch, capsys) -> None:
    """The printed PLAN is rendered from the argv that later runs, so a
    truthful argv makes a truthful plan — the issue's point 1."""
    _scaffold_fleet(tmp_path)
    assert main(["up", "cortex", "--compose-dir", str(tmp_path), "--json"]) == 0
    planned = json.loads(capsys.readouterr().out)["command"]

    captured: dict = {}
    monkeypatch.setattr(
        _compose, "run_compose", lambda d, argv: (captured.setdefault("argv", argv), _ok())[1]
    )
    assert main(["up", "cortex", "--compose-dir", str(tmp_path), "--apply", "--json"]) == 0
    assert " ".join(captured["argv"]) == planned
    assert _compose.NO_DEPS_FLAG in planned


# --- guard 2: nothing makes a core lane a dependency of the gateway ---------


def _fleet_template() -> dict:
    text = files("lobes.templates.fleet").joinpath("docker-compose.yml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def test_gateway_declares_no_depends_on() -> None:
    """The regression this issue is: the gateway listed all four core lanes as
    `depends_on`, so ANY gateway start walked into starting all four — a
    profile/feasibility declaration cannot override a dependency edge."""
    assert "depends_on" not in _fleet_template()["services"]["gateway"]


def test_no_core_generate_lane_is_anyone_s_hard_dependency() -> None:
    """The heavy lanes may DEPEND on the two cheap pooling gears (a measured
    unified-memory boot-ordering mitigation, docs/machine-profiles.md), but
    nothing may depend on THEM: those are the lanes a shape drops, and a
    dependency edge would resurrect a dropped one."""
    heavy = {
        "vllm-primary",
        "vllm-multimodal",
        "vllm-muse",
        "vllm-worker",
        "vllm-associate",
    }
    for name, svc in _fleet_template()["services"].items():
        depends = svc.get("depends_on") or {}
        named = set(depends if isinstance(depends, dict) else depends)
        assert not (named & heavy), f"{name!r} depends on a droppable heavy lane: {named & heavy}"


# --- the gateway update path (issue #222 point 3) ---------------------------


def test_up_gateway_targets_only_the_gateway(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    assert main(["up", "gateway", "--compose-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["services"] == ["gateway"]
    assert payload["command"] == "docker compose up -d --no-deps gateway"
    assert payload["build"] is False


def test_up_gateway_build_re_images_only_the_gateway(tmp_path, capsys) -> None:
    """The operator need that produced this issue: reinstall the gateway at a
    new MODEL_GEAR_VERSION, touch nothing else."""
    _scaffold_fleet(tmp_path)
    assert main(["up", "gateway", "--build", "--compose-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "docker compose up -d --no-deps --build gateway"
    assert payload["build"] is True


def test_up_gateway_never_pulls_in_the_audio_overlay(tmp_path, capsys) -> None:
    """The gateway FRONTS the audio lanes over HTTP; it does not declare them,
    so a gateway restart must not reach into the overlay."""
    _scaffold_fleet_audio(tmp_path)
    assert main(["up", "gateway", "--compose-dir", str(tmp_path), "--json"]) == 0
    assert "docker-compose.audio.yml" not in json.loads(capsys.readouterr().out)["command"]


def test_up_gateway_down_stops_only_the_gateway(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    assert main(["up", "gateway", "--down", "--compose-dir", str(tmp_path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "docker compose stop gateway"


def test_build_with_down_is_a_user_error(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    assert main(["up", "gateway", "--down", "--build", "--compose-dir", str(tmp_path)]) != 0
    assert "--build has no meaning with --down" in capsys.readouterr().err


def test_gateway_is_a_target_but_not_a_role() -> None:
    """It must never leak into the role registry, the colleague-stack bundle,
    or anything else that enumerates Colleague roles."""
    from lobes import roles

    assert up_cmd.GATEWAY_TARGET in up_cmd.TARGETS
    assert up_cmd.GATEWAY_TARGET not in roles.ROLES
    assert up_cmd.GATEWAY_TARGET not in up_cmd.ROLE_SERVICE
    stack, _ = up_cmd._resolve(up_cmd.COLLEAGUE_STACK)
    assert up_cmd.GATEWAY_SERVICE not in stack
