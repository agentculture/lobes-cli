"""``lobes up <role> --replace`` and the memory gate.

The Spark card declares ``cortex`` and ``innereye`` exclusive (one unified
memory pool). These tests drive ``lobes up`` against a scaffolded fleet whose
``.env`` names that card, with container state, ``/proc/meminfo`` and the
compose runner all faked. No docker.
"""

from __future__ import annotations

import json
import types

import pytest

from lobes.cli import _role_swap, main
from lobes.runtime import _compose

_GIB_KB = 1024 * 1024


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


@pytest.fixture
def box(tmp_path, monkeypatch):
    """A scaffolded Spark fleet, cortex running, plenty of memory, compose faked."""
    _compose.write_scaffold(tmp_path, force=True, templates=_compose.FLEET_TEMPLATES)
    (tmp_path / ".env").write_text(
        "LOBES_PROFILE=spark\nPRIMARY_GPU_MEM_UTIL=0.58\nPRIMARY_FEASIBLE=true\n"
        "COMPOSE_PROFILES=\nINNEREYE_UI_PORT=0.0.0.0:8188\n",
        encoding="utf-8",
    )
    state = {"running": {"model-gear-vllm-primary"}, "calls": []}
    monkeypatch.setattr(
        _compose,
        "container_status",
        lambda name: "running" if name in state["running"] else "exited",
    )

    def fake_run(deploy_dir, argv):
        state["calls"].append(argv)
        svc = argv[-1]
        if "stop" in argv:
            state["running"].discard("model-gear-" + svc)
        elif "up" in argv:
            state["running"].add("model-gear-" + svc)
        return _ok()

    monkeypatch.setattr(_compose, "run_compose", fake_run)
    monkeypatch.setattr("lobes.cli._commands.up.trigger_reannounce", lambda *a, **k: None)
    meminfo = tmp_path / "meminfo"
    state["meminfo"] = meminfo

    def set_mem(total_gib: float, available_gib: float) -> None:
        meminfo.write_text(
            f"MemTotal: {int(total_gib * _GIB_KB)} kB\n"
            f"MemAvailable: {int(available_gib * _GIB_KB)} kB\n",
            encoding="utf-8",
        )

    state["set_mem"] = set_mem
    set_mem(121.7, 80.0)
    monkeypatch.setattr(_role_swap, "MEMINFO", meminfo)
    monkeypatch.setattr(_role_swap, "RELEASE_WAIT_S", 0.0)
    state["dir"] = tmp_path
    return state


def _env(box) -> str:
    return (box["dir"] / ".env").read_text(encoding="utf-8")


def _up(box, *args: str) -> int:
    return main(["up", *args, "--compose-dir", str(box["dir"])])


# --- the exclusivity guard ---------------------------------------------------


def test_up_innereye_beside_running_cortex_is_refused_naming_replace(box, capsys) -> None:
    assert _up(box, "innereye", "--apply") != 0
    err = capsys.readouterr().err
    assert "can't run beside cortex" in err
    assert "lobes up innereye --replace --apply" in err
    assert box["calls"] == []


def test_up_cortex_beside_running_innereye_is_refused(box, capsys) -> None:
    box["running"] = {"model-gear-comfyui"}
    assert _up(box, "cortex", "--apply") != 0
    assert "can't run beside innereye" in capsys.readouterr().err


def test_up_with_the_rival_stopped_needs_no_replace(box, capsys) -> None:
    """cortex already down: plain `lobes up innereye` is just the old gate."""
    box["running"] = set()
    (box["dir"] / ".env").write_text(
        _env(box).replace("COMPOSE_PROFILES=", "COMPOSE_PROFILES=innereye"), encoding="utf-8"
    )
    assert _up(box, "innereye", "--json") == 0
    assert json.loads(capsys.readouterr().out)["services"] == ["comfyui"]


def test_replace_on_a_role_with_no_rival_is_a_user_error(box, capsys) -> None:
    assert _up(box, "hand", "--replace") != 0
    assert "nothing to replace" in capsys.readouterr().err


def test_replace_with_down_is_a_user_error(box, capsys) -> None:
    assert _up(box, "innereye", "--replace", "--down") != 0
    assert "--replace has no meaning with --down" in capsys.readouterr().err


# --- --replace ---------------------------------------------------------------


def test_replace_dry_run_lists_the_whole_switch_and_changes_nothing(box, capsys) -> None:
    before = _env(box)
    assert _up(box, "innereye", "--replace") == 0
    out = capsys.readouterr().out
    assert "stop vllm-primary" in out
    assert ".env: PRIMARY_FEASIBLE=false" in out
    assert ".env: COMPOSE_PROFILES=innereye" in out
    assert ".env: INNEREYE_BASE_URL=http://comfyui:8188" in out
    assert "up -d --no-deps comfyui" in out
    assert "up -d --no-deps gateway" in out
    assert box["calls"] == []
    assert _env(box) == before


def test_replace_apply_switches_cortex_to_innereye(box) -> None:
    assert _up(box, "innereye", "--replace", "--apply") == 0
    calls = [" ".join(c) for c in box["calls"]]
    assert calls[0].endswith("stop vllm-primary")
    assert calls[1].endswith("up -d --no-deps comfyui")
    assert calls[2].endswith("up -d --no-deps gateway")
    env = _env(box)
    assert "PRIMARY_FEASIBLE=false" in env
    assert "INNEREYE_FEASIBLE=true" in env
    assert "INNEREYE_BASE_URL=http://comfyui:8188" in env
    assert "COMPOSE_PROFILES=innereye" in env
    assert list(box["dir"].glob(".env.bak-*-switch-to-innereye"))


def test_replace_back_to_cortex_reverses_the_env(box) -> None:
    assert _up(box, "innereye", "--replace", "--apply") == 0
    box["calls"].clear()
    assert _up(box, "cortex", "--replace", "--apply") == 0
    calls = [" ".join(c) for c in box["calls"]]
    # comfyui is profile-gated: the stop must carry its profile to see it.
    assert "--profile innereye" in calls[0] and calls[0].endswith("stop comfyui")
    env = _env(box)
    assert "PRIMARY_FEASIBLE=true" in env
    assert "INNEREYE_FEASIBLE=false" in env
    assert "COMPOSE_PROFILES=\n" in env


# --- the memory gate ---------------------------------------------------------


def test_short_memory_refuses_and_names_override(box, capsys) -> None:
    box["running"] = set()
    (box["dir"] / ".env").write_text(
        _env(box).replace("COMPOSE_PROFILES=", "COMPOSE_PROFILES=innereye"), encoding="utf-8"
    )
    box["set_mem"](121.7, 20.0)
    assert _up(box, "innereye", "--apply") != 0
    err = capsys.readouterr().err
    assert "not enough memory" in err and "31.4 GiB" in err
    assert "--override-memory" in err
    assert box["calls"] == []


def test_override_memory_starts_anyway(box) -> None:
    box["running"] = set()
    (box["dir"] / ".env").write_text(
        _env(box).replace("COMPOSE_PROFILES=", "COMPOSE_PROFILES=innereye"), encoding="utf-8"
    )
    box["set_mem"](121.7, 20.0)
    assert _up(box, "innereye", "--apply", "--override-memory") == 0
    assert box["calls"][-1][-1] == "comfyui"


def test_vllm_role_needs_its_util_share_of_total(box, capsys) -> None:
    """cortex declares no peak; its requirement is PRIMARY_GPU_MEM_UTIL x MemTotal."""
    box["running"] = set()
    box["set_mem"](121.7, 40.0)
    assert _up(box, "cortex", "--apply") != 0
    assert "PRIMARY_GPU_MEM_UTIL=0.58" in capsys.readouterr().err


def test_an_already_running_role_is_not_memory_gated(box) -> None:
    box["set_mem"](121.7, 5.0)
    assert _up(box, "cortex", "--apply") == 0


def test_replace_short_after_stop_restarts_the_rival_and_leaves_env(box, capsys) -> None:
    before = _env(box)
    box["set_mem"](121.7, 10.0)
    assert _up(box, "innereye", "--replace", "--apply") != 0
    calls = [" ".join(c) for c in box["calls"]]
    assert calls[0].endswith("stop vllm-primary")
    assert calls[-1].endswith("up -d --no-deps vllm-primary")
    assert "model-gear-comfyui" not in box["running"]
    assert _env(box) == before
    assert "restarted cortex" in capsys.readouterr().err


def test_replace_with_override_memory_skips_the_gate(box) -> None:
    box["set_mem"](121.7, 10.0)
    assert _up(box, "innereye", "--replace", "--apply", "--override-memory") == 0
    assert "COMPOSE_PROFILES=innereye" in _env(box)


def test_unreadable_meminfo_does_not_block(box, monkeypatch) -> None:
    box["running"] = set()
    monkeypatch.setattr(_role_swap, "MEMINFO", box["dir"] / "missing")
    (box["dir"] / ".env").write_text(
        _env(box).replace("COMPOSE_PROFILES=", "COMPOSE_PROFILES=innereye"), encoding="utf-8"
    )
    assert _up(box, "innereye", "--apply") == 0
