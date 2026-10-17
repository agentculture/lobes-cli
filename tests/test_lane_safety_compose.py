"""Lane-safety t7: every compose service runs the boot hold (mg-logwrap).

Contract: docs/contracts/lane-safety-runtime.md sections 1-2. Every service in
every packaged compose template, and every rendered decider sidecar, must

* run ``mg-logwrap`` as its entrypoint (the wrapper implements the hold and
  then ``exec``s the real command, which is respelled as ``command:``),
* receive ``LOBES_BOOT_DELAY_S`` / ``LOBES_BOOT_ORDER`` / ``LOBES_BOOT_STAGGER_S``
  (``env_file: .env`` or an explicit ``environment:`` passthrough),
* carry a healthcheck ``start_period`` that covers the longest hold, and
* keep ``restart: unless-stopped``.
"""

from __future__ import annotations

import re
from importlib.resources import files

import pytest
import yaml

from lobes.cli._commands import init as init_cmd
from lobes.profiles.loader import resolve_profile
from lobes.profiles.shapes import resolve_shape

KNOBS = ("LOBES_BOOT_DELAY_S", "LOBES_BOOT_ORDER", "LOBES_BOOT_STAGGER_S")
WRAPPER = ["bash", "/usr/local/bin/mg-logwrap"]
WRAPPER_MOUNT = "./mg-logwrap.sh:/usr/local/bin/mg-logwrap:ro"
# Max hold with the shipped defaults: delay 120 + an unlisted lane at
# len(order) * stagger <= 3 * 20 (orin's "multimodal,embed,rerank").
MAX_HOLD_S = 180

TEMPLATES = (
    "docker-compose.yml",
    "fleet/docker-compose.yml",
    "fleet/docker-compose.audio.yml",
    "fleet/docker-compose.audio-he.yml",
    "fleet/docker-compose.embed.yml",
)

# Pre-hold healthcheck start_period (seconds) per service. The held lane must
# declare at least this plus MAX_HOLD_S. None means the file adds nothing for
# that key (a deep-merge overlay that inherits it from the base overlay).
BASE_START_PERIOD = {
    "vllm": 600,
    "llamacpp-primary": 600,
    "comfyui": 90,
    "gateway": 20,
    "chatterbox": 120,
    "stt": 600,
    "realtime": 30,
    "embed-gemma2-embed": 300,
    "embed-qwen3vl-embed": 900,
    "embed-qwen3vl-rerank": 900,
    "embed-nemotron-embed": 900,
    "embed-nomic-code-embed": 900,
    "decider": 900,
}


def _load(rel: str) -> dict:
    return yaml.safe_load((files("lobes.templates") / rel).read_text())


def _seconds(value: str) -> int:
    m = re.fullmatch(r"(?:\$\{[A-Z0-9_]+:-)?(\d+)s\}?", str(value))
    assert m, f"unparseable duration {value!r}"
    return int(m.group(1))


def _env_keys(svc: dict) -> set[str]:
    keys: set[str] = set()
    env = svc.get("environment") or []
    if isinstance(env, dict):
        keys |= set(env)
    else:
        keys |= {e.split("=", 1)[0] for e in env}
    return keys


def _has_env_file(svc: dict) -> bool:
    return any(
        (e == ".env" if isinstance(e, str) else e.get("path") == ".env")
        for e in svc.get("env_file") or []
    )


def _all_services():
    for rel in TEMPLATES:
        for name, svc in (_load(rel).get("services") or {}).items():
            yield pytest.param(rel, name, svc, id=f"{rel}:{name}")


def _decider_services() -> dict:
    shape = resolve_shape("orin-decider")
    text = init_cmd.render_shape_override(shape, resolve_profile("orin"))
    assert text is not None
    services = yaml.safe_load(text)["services"]
    return {k: v for k, v in services.items() if k.startswith("decider-")}


def _check_held(name: str, svc: dict, merged_with: dict | None = None) -> None:
    merged = {**(merged_with or {}), **svc}
    assert merged["entrypoint"] == WRAPPER, f"{name}: no mg-logwrap entrypoint"
    assert merged.get("command"), f"{name}: real command not respelled after the wrapper"
    assert WRAPPER_MOUNT in merged["volumes"], f"{name}: wrapper not mounted"
    assert any(v.endswith(":/logs/model-gear") for v in merged["volumes"]), name
    env = _env_keys(svc) | (_env_keys(merged_with) if merged_with else set())
    assert "MG_LOG_NAME" in env, f"{name}: no MG_LOG_NAME"
    if not _has_env_file(merged):
        for knob in KNOBS:
            assert knob in env, f"{name}: {knob} not passed through"


def test_services_are_discovered() -> None:
    names = {p.values[1] for p in _all_services()}
    assert {"vllm", "vllm-primary", "gateway", "chatterbox", "stt", "realtime"} <= names
    assert "embed-gemma2-embed" in names


@pytest.mark.parametrize("rel,name,svc", _all_services())
def test_every_service_runs_the_hold(rel: str, name: str, svc: dict) -> None:
    if rel.endswith("audio-he.yml") and name in ("gateway", "chatterbox", "realtime", "stt"):
        # Overlay of the English audio file: wrapper/knobs are inherited via
        # merge; only stt swaps the real command (checked below).
        pytest.skip("inherits the hold from docker-compose.audio.yml")
    if rel.endswith("embed.yml") and name == "gateway":
        pytest.skip("env-only override merged onto the base gateway")
    if rel.endswith("audio.yml") and name == "gateway":
        pytest.skip("env-only override merged onto the base gateway")
    _check_held(name, svc)


def test_hebrew_stt_respells_the_whisper_command() -> None:
    he = _load("fleet/docker-compose.audio-he.yml")["services"]["stt"]
    en = _load("fleet/docker-compose.audio.yml")["services"]["stt"]
    assert "listen_server_whisper.py" in " ".join(he["command"])
    assert "listen_server.py" in " ".join(en["command"])
    assert "entrypoint" not in he  # inherited from the English overlay
    he_ml = _load("fleet/docker-compose.audio-he.yml")["services"]["chatterbox"]
    assert "chatterbox_multilingual_server" in " ".join(he_ml["command"])


def test_real_commands_match_the_dockerfiles() -> None:
    """Respelled commands equal the image ENTRYPOINT/CMD they replace."""
    audio = _load("fleet/docker-compose.audio.yml")["services"]
    fleet = _load("fleet/docker-compose.yml")["services"]
    assert audio["chatterbox"]["command"][-2:] == ["-m", "lobes.realtime.chatterbox_server"]
    assert audio["realtime"]["command"] == ["python", "-m", "lobes.realtime"]
    assert fleet["gateway"]["command"] == ["python", "-m", "lobes.gateway"]
    assert audio["stt"]["command"] == ["python3", "/app/listen_server.py"]
    embed = _load("fleet/docker-compose.embed.yml")["services"]["embed-gemma2-embed"]
    assert embed["command"] == ["python3", "-m", "lobes.embed_sidecar.server"]


def test_decider_sidecars_run_the_hold() -> None:
    services = _decider_services()
    assert services
    for name, svc in services.items():
        _check_held(name, svc)
        assert svc["command"] == ["python3", "-m", "lobes.decider_sidecar.server"]
        assert svc["restart"] == "unless-stopped"
        env = dict(e.split("=", 1) for e in svc["environment"])
        assert env["MG_LOG_NAME"] == name
        assert _seconds(svc["healthcheck"]["start_period"]) >= (
            BASE_START_PERIOD["decider"] + MAX_HOLD_S
        )


# Entries that only merge onto a service defined in an earlier file.
OVERLAY_ONLY = {
    ("fleet/docker-compose.audio.yml", "gateway"),
    ("fleet/docker-compose.embed.yml", "gateway"),
    ("fleet/docker-compose.audio-he.yml", "stt"),
    ("fleet/docker-compose.audio-he.yml", "chatterbox"),
    ("fleet/docker-compose.audio-he.yml", "gateway"),
    ("fleet/docker-compose.audio-he.yml", "realtime"),
}


@pytest.mark.parametrize("rel,name,svc", _all_services())
def test_start_period_covers_the_hold_and_restart_unchanged(rel: str, name: str, svc: dict) -> None:
    if (rel, name) in OVERLAY_ONLY:
        assert svc.get("restart", "unless-stopped") == "unless-stopped"
        return
    assert svc.get("restart") == "unless-stopped", f"{name}: restart policy changed"
    start = (svc.get("healthcheck") or {}).get("start_period")
    assert start, f"{name}: no healthcheck.start_period"
    base = BASE_START_PERIOD.get(name, 600)
    assert _seconds(start) >= base + MAX_HOLD_S, f"{name}: {start} < {base}+{MAX_HOLD_S}"


def test_he_chatterbox_start_period_covers_hold() -> None:
    he = _load("fleet/docker-compose.audio-he.yml")["services"]["chatterbox"]
    assert _seconds(he["healthcheck"]["start_period"]) >= 300 + MAX_HOLD_S


# --- t8: per-lane vLLM compile cache (contract section 6) -------------------

CACHE_ENV = "VLLM_CACHE_ROOT=${LOBES_VLLM_CACHE_ROOT:-/cache/vllm}"
CACHE_TARGET = "/cache/vllm"


def _is_vllm_service(svc: dict) -> bool:
    cmd = svc.get("command")
    if isinstance(cmd, list):
        cmd = " ".join(str(c) for c in cmd)
    return "vllm serve" in " ".join(str(cmd or "").split()) or (
        isinstance(svc.get("command"), list) and svc["command"][:2] == ["vllm", "serve"]
    )


def _lane(svc: dict) -> str:
    for e in svc["environment"]:
        if e.startswith("MG_LOG_NAME="):
            return e.split("=", 1)[1]
    raise AssertionError("no MG_LOG_NAME")


def _cache_volume(lane: str) -> str:
    return f"${{LOBES_COMPILE_CACHE_DIR:-./compile-cache}}/{lane}:{CACHE_TARGET}"


@pytest.mark.parametrize("rel,name,svc", _all_services())
def test_compile_cache_mount_only_on_vllm_services(rel: str, name: str, svc: dict) -> None:
    vols = svc.get("volumes") or []
    has_mount = any(v.endswith(f":{CACHE_TARGET}") for v in vols)
    has_env = CACHE_ENV in (svc.get("environment") or [])
    if _is_vllm_service(svc):
        assert has_mount and has_env, f"{name}: vLLM lane without per-lane compile cache"
        assert _cache_volume(_lane(svc)) in vols
    else:
        assert not has_mount and not has_env, f"{name}: non-vLLM service has a compile cache"


def test_every_expected_vllm_lane_is_covered() -> None:
    found = {
        name for rel, name, svc in (p.values for p in _all_services()) if _is_vllm_service(svc)
    }
    assert found >= {
        "vllm",
        "vllm-primary",
        "vllm-embed",
        "vllm-embed-deep",
        "vllm-rerank",
        "vllm-minor",
        "vllm-hand",
        "vllm-middle",
        "vllm-multimodal",
        "vllm-multimodal-coder",
        "vllm-muse",
        "vllm-worker",
        "vllm-associate",
        "embed-qwen3vl-embed",
        "embed-qwen3vl-rerank",
        "embed-nemotron-embed",
        "embed-nomic-code-embed",
    }
    assert "llamacpp-primary" not in found and "embed-gemma2-embed" not in found


def test_clef_vllm_decider_has_the_cache_and_python_decider_does_not() -> None:
    from lobes.profiles.shape_render import _decider_service_lines

    clef = _decider_service_lines(
        "general", {"backend": "clef-vllm", "model": "m", "mem_limit": "1g"}, "runtime"
    )
    text = "\n".join(clef)
    assert CACHE_ENV in text
    assert "/decider-general:/cache/vllm" in text
    other = "\n".join(
        _decider_service_lines(
            "general", {"backend": "clef", "model": "m", "mem_limit": "1g"}, "runtime"
        )
    )
    assert "VLLM_CACHE_ROOT" not in other and "/cache/vllm" not in other


def test_off_value_is_off_the_mount_target() -> None:
    """The opt-out value must differ from the fixed mount target, else the bind
    would still capture the cache (the writable layer would not be restored)."""
    assert CACHE_ENV.endswith(":-/cache/vllm}")
    assert "/root/.cache/vllm" != CACHE_TARGET
