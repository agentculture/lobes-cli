"""Tests for the ``lobes fleet`` verbs (up / down / status) and ``init --fleet``."""

from __future__ import annotations

import json
import types

from lobes.cli import main
from lobes.runtime import _compose, _detect, _health


def _ok() -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=0, stdout="", stderr="")


def _scaffold_fleet(path):
    _compose.write_scaffold(path, force=True, templates=_compose.FLEET_TEMPLATES)
    return path


def _scaffold_fleet_audio(path):
    templates = {**_compose.FLEET_TEMPLATES, **_compose.AUDIO_TEMPLATES}
    _compose.write_scaffold(path, force=True, templates=templates)
    return path


def _write_shape_override(path, dropped_services):
    """Write a minimal shape override that parks ``dropped_services`` in the inert
    profile — mirrors the exact format ``lobes init`` generates (see
    tests/test_init_shape.py for the round-trip against the real generator)."""
    lines = ["services:"]
    for svc in dropped_services:
        lines.append(f"  {svc}:")
        lines.append('    profiles: ["shape-dropped"]')
    lines.append("  gateway:")
    lines.append("    depends_on: !reset null")
    (path / _compose.SHAPE_OVERLAY).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# --- fleet up -------------------------------------------------------------


def test_fleet_up_dry_run_changes_nothing(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)

    def boom(*a, **k):
        raise AssertionError("compose ran during dry-run")

    monkeypatch.setattr(_compose, "compose_up_build", boom)
    rc = main(["fleet", "up", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_fleet_up_apply_builds_and_waits(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(
        _compose, "compose_up_build", lambda d: (calls.append("up-build"), _ok())[1]
    )
    waited: dict = {}

    def fake_wait(port, **kw):
        waited["port"] = port
        waited["container"] = kw.get("container")

    monkeypatch.setattr(_health, "wait_health", fake_wait)
    rc = main(["fleet", "up", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    assert calls == ["up-build"]
    assert waited["container"] == _compose.FLEET_GATEWAY  # waits on the gateway front


# --- fleet down -----------------------------------------------------------


def test_fleet_down_dry_run(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["fleet", "down", "--compose-dir", str(tmp_path)])
    assert rc == 0
    assert "DRY RUN" in capsys.readouterr().out


def test_fleet_down_apply(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    calls: list[str] = []
    monkeypatch.setattr(_compose, "compose_down", lambda d: (calls.append("down"), _ok())[1])
    rc = main(["fleet", "down", "--compose-dir", str(tmp_path), "--apply"])
    assert rc == 0
    assert calls == ["down"]


# --- fleet status ---------------------------------------------------------


def test_fleet_status_json_reports_default_containers(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    names = [c["name"] for c in payload["containers"]]
    # The default fleet runs five containers: generate primary, multimodal (always-on),
    # embedding + reranker gears, and the gateway (the generate fallback is opt-in, excluded).
    assert names == list(_compose.FLEET_CONTAINERS)
    assert names == [
        "model-gear-vllm-primary",
        "model-gear-vllm-multimodal",
        "model-gear-vllm-embed",
        "model-gear-vllm-rerank",
        "model-gear-gateway",
    ]
    # offline fixture: _probe → None (state "not created"), is_healthy → False.
    assert all(c["state"] == "not created" for c in payload["containers"])
    assert payload["gateway_health"] == "not responding"
    assert payload["models"] is None  # not healthy → no /v1/models fetch
    assert payload["port"] == 8000


def test_bare_fleet_defaults_to_status(tmp_path, capsys) -> None:
    _scaffold_fleet(tmp_path)
    rc = main(["fleet", "--compose-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "gateway:" in out
    assert _compose.FLEET_GATEWAY in out


def test_fleet_status_fetches_models_when_healthy(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet(tmp_path)
    monkeypatch.setattr(_health, "is_healthy", lambda *a, **k: True)
    from lobes import assess

    monkeypatch.setattr(
        assess, "_get", lambda url, path, timeout=10: (200, {"data": [{"id": "P"}, {"id": "F"}]})
    )
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["gateway_health"] == "ok"
    assert payload["models"] == ["P", "F"]


def test_fleet_status_unscaffolded_errors(capsys) -> None:
    # No deployment scaffolded (autouse fixture points the home at an empty dir).
    rc = main(["fleet", "status"])
    assert rc == 2  # EXIT_ENV_ERROR
    assert "hint:" in capsys.readouterr().err


# --- audio overlay awareness ----------------------------------------------


def test_compose_files_only_adds_overlay_when_present(tmp_path) -> None:
    _scaffold_fleet(tmp_path)  # no audio overlay
    assert _compose._compose_files(tmp_path) == []
    assert _compose.audio_overlay_present(tmp_path) is False
    _scaffold_fleet_audio(tmp_path)  # now with the overlay
    assert _compose.audio_overlay_present(tmp_path) is True
    assert _compose._compose_files(tmp_path) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.AUDIO_OVERLAY,
    ]


def test_compose_up_build_includes_overlay_argv(tmp_path, monkeypatch) -> None:
    _scaffold_fleet_audio(tmp_path)
    captured: dict = {}
    monkeypatch.setattr(
        _compose, "_run", lambda argv, **kw: captured.setdefault("argv", argv) or _ok()
    )
    _compose.compose_up_build(tmp_path)
    assert captured["argv"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.audio.yml",
        "up",
        "-d",
        "--build",
    ]


def test_fleet_status_includes_audio_containers_with_overlay(tmp_path, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    names = [c["name"] for c in payload["containers"]]
    assert names == list(_compose.FLEET_CONTAINERS) + list(_compose.FLEET_AUDIO_CONTAINERS)


def test_audio_container_constants_match_compose_container_names() -> None:
    """The FLEET_AUDIO_CONTAINERS constants must equal the `container_name:`
    entries in the packaged audio compose, or `lobes fleet status` reports a gear
    as "not created" (the Magpie->Chatterbox rename drifted FLEET_TTS once)."""
    from importlib.resources import files

    overlay = (files("lobes.templates") / "fleet" / "docker-compose.audio.yml").read_text(
        encoding="utf-8"
    )
    declared = {
        line.split("container_name:", 1)[1].strip()
        for line in overlay.splitlines()
        if "container_name:" in line
    }
    for name in _compose.FLEET_AUDIO_CONTAINERS:
        assert name in declared, f"{name} has no matching container_name in the audio compose"


def test_chatterbox_healthcheck_uses_image_python() -> None:
    """The chatterbox image installs `python3.12` with no `python3` symlink, so
    the healthcheck must invoke `python3.12`. A bare `python3` exec-fails forever
    and pins the (working) container at "starting"/"unhealthy"."""
    import re
    from importlib.resources import files

    root = files("lobes.templates") / "fleet"
    overlay = (root / "docker-compose.audio.yml").read_text(encoding="utf-8").splitlines()
    # isolate the `  chatterbox:` service block (up to the next 2-space service key)
    start = next(i for i, ln in enumerate(overlay) if ln.startswith("  chatterbox:"))
    end = next(
        (i for i in range(start + 1, len(overlay)) if re.match(r"  \S", overlay[i])),
        len(overlay),
    )
    block = "\n".join(overlay[start:end])
    assert "- python3.12" in block, "chatterbox healthcheck must call python3.12"
    assert "\n        - python3\n" not in block, "bare python3 isn't on PATH in the image"
    # ...and that interpreter is the one the Dockerfile actually provides.
    assert "python3.12" in (root / "Dockerfile.chatterbox").read_text(encoding="utf-8")


def test_fleet_up_reports_audio_containers_with_overlay(tmp_path, monkeypatch, capsys) -> None:
    _scaffold_fleet_audio(tmp_path)
    monkeypatch.setattr(_compose, "compose_up_build", lambda d: _ok())
    monkeypatch.setattr(_health, "wait_health", lambda *a, **k: None)
    rc = main(["fleet", "up", "--compose-dir", str(tmp_path), "--apply", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["containers"] == (
        list(_compose.FLEET_CONTAINERS) + list(_compose.FLEET_AUDIO_CONTAINERS)
    )


# --- shape overlay awareness (brain-shapes t4b) ------------------------------


def test_shape_overlay_present_detects_the_file(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    assert _compose.shape_overlay_present(tmp_path) is False
    _write_shape_override(tmp_path, ["vllm-multimodal"])
    assert _compose.shape_overlay_present(tmp_path) is True


def test_compose_files_includes_shape_overlay_after_base(tmp_path) -> None:
    """Acceptance 4: the -f chain gains the shape overlay (after the base) when present."""
    _scaffold_fleet(tmp_path)  # no overlays
    assert _compose._compose_files(tmp_path) == []
    _write_shape_override(tmp_path, ["vllm-multimodal"])
    assert _compose._compose_files(tmp_path) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.SHAPE_OVERLAY,
    ]


def test_compose_files_orders_shape_after_audio(tmp_path) -> None:
    """Both overlays present → base, then audio, then shape LAST (so the shape's
    !reset on gateway.depends_on is applied after everything else)."""
    _scaffold_fleet_audio(tmp_path)
    _write_shape_override(tmp_path, ["vllm-primary"])
    assert _compose._compose_files(tmp_path) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.AUDIO_OVERLAY,
        "-f",
        _compose.SHAPE_OVERLAY,
    ]


def test_compose_up_build_includes_shape_overlay_argv(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    _write_shape_override(tmp_path, ["vllm-multimodal"])
    captured: dict = {}
    monkeypatch.setattr(
        _compose, "_run", lambda argv, **kw: captured.setdefault("argv", argv) or _ok()
    )
    _compose.compose_up_build(tmp_path)
    assert captured["argv"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.shape.yml",
        "up",
        "-d",
        "--build",
    ]


def test_compose_down_includes_shape_overlay_argv(tmp_path, monkeypatch) -> None:
    _scaffold_fleet(tmp_path)
    _write_shape_override(tmp_path, ["vllm-primary"])
    captured: dict = {}
    monkeypatch.setattr(
        _compose, "_run", lambda argv, **kw: captured.setdefault("argv", argv) or _ok()
    )
    _compose.compose_down(tmp_path)
    assert captured["argv"] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.shape.yml",
        "down",
    ]


def test_shape_dropped_containers_reads_the_override(tmp_path) -> None:
    _scaffold_fleet(tmp_path)
    assert _compose.shape_dropped_containers(tmp_path) == ()
    _write_shape_override(tmp_path, ["vllm-multimodal"])
    assert _compose.shape_dropped_containers(tmp_path) == (_compose.FLEET_MULTIMODAL,)


def test_fleet_containers_excludes_shape_dropped_service(tmp_path) -> None:
    """Acceptance 5: the expected container list drops the shape-disabled gear."""
    _scaffold_fleet(tmp_path)
    _write_shape_override(tmp_path, ["vllm-multimodal"])
    containers = _compose.fleet_containers(tmp_path)
    assert _compose.FLEET_MULTIMODAL not in containers
    assert _compose.FLEET_PRIMARY in containers
    assert _compose.FLEET_EMBED in containers
    assert _compose.FLEET_RERANK in containers
    assert _compose.FLEET_GATEWAY in containers


def test_fleet_status_excludes_dropped_container(tmp_path, capsys) -> None:
    """Acceptance 5 end-to-end: `lobes fleet status` omits the dropped gear."""
    _scaffold_fleet(tmp_path)
    _write_shape_override(tmp_path, ["vllm-primary"])
    rc = main(["fleet", "status", "--compose-dir", str(tmp_path), "--json"])
    assert rc == 0
    names = [c["name"] for c in json.loads(capsys.readouterr().out)["containers"]]
    assert _compose.FLEET_PRIMARY not in names
    assert _compose.FLEET_MULTIMODAL in names


def test_fleet_containers_matches_real_init_generated_override(tmp_path, monkeypatch) -> None:
    """Ties the reader to the real generator: an init-scaffolded spark-lobe override
    excludes exactly model-gear-vllm-multimodal from the fleet set."""
    monkeypatch.setattr(_detect, "detect_card", lambda: _fake_card("spark"))
    target = tmp_path / "deploy"
    assert main(["init", "--shape", "spark-lobe", str(target), "--apply"]) == 0
    assert _compose.shape_dropped_containers(target) == (_compose.FLEET_MULTIMODAL,)
    assert _compose.FLEET_MULTIMODAL not in _compose.fleet_containers(target)


def _fake_card(resolved: str) -> _detect.DetectedCard:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name="NVIDIA GB10",
        compute_capability="sm_121",
        total_memory_gb=119.7,
        hostname="test-host",
        device_tree_model=None,
        sources={},
    )


# --- fleet compose template assertions (vllm-multimodal default-on) ----------

_GEMMA_CODER_ID = "sakamakismile/gemma-4-12B-coder-fable5-composer2.5-MTP-NVFP4"


def _fleet_compose_text() -> str:
    from importlib.resources import files

    return (files("lobes.templates") / "fleet" / "docker-compose.yml").read_text(encoding="utf-8")


def _service_block(text: str, service_name: str) -> str:
    """Extract the YAML block for a top-level service (until the next 2-space key)."""
    import re

    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if re.match(rf"  {re.escape(service_name)}:", ln)),
        None,
    )
    assert start is not None, f"service '{service_name}' not found in fleet compose"
    end = next(
        (i for i in range(start + 1, len(lines)) if re.match(r"  \S", lines[i])),
        len(lines),
    )
    return "\n".join(lines[start:end])


def test_fleet_compose_multimodal_is_default_on() -> None:
    """vllm-multimodal service exists and has NO profiles: key — starts by default."""
    text = _fleet_compose_text()
    assert "vllm-multimodal:" in text, "vllm-multimodal service must be defined in fleet compose"
    block = _service_block(text, "vllm-multimodal")
    assert (
        "profiles:" not in block
    ), "vllm-multimodal must not have a profiles: key — it must come up with the default fleet"


def test_fleet_compose_multimodal_vision_active_has_native_mtp_spec_decode() -> None:
    """vllm-multimodal: no --language-model-only (vision+audio active); HAS --speculative-config.

    "Support both" (docs/vllm-nightly-migration.md §7, 2026-07-02): the default
    "multimodal" gear is now the NVFP4 BASE it-model wired to the public
    google/gemma-4-12B-it-assistant native-MTP draft — measured 28.6 tok/s decode
    at 57.9% draft acceptance, the fastest Gemma config on this hardware. The
    exact JSON must match lobes.catalog.SUPPORTED_MODELS' coolthor entry (guarded
    by tests/test_catalog.py's round-trip test).
    """
    block = _service_block(_fleet_compose_text(), "vllm-multimodal")
    assert (
        "--language-model-only" not in block
    ), "vllm-multimodal must NOT pass --language-model-only: vision+audio must stay active"
    assert "--speculative-config" in block, (
        "vllm-multimodal must carry --speculative-config: native MTP is default-on for "
        "the NVFP4 base gear (§7, 28.6 tok/s @ 57.9% draft acceptance)"
    )
    assert (
        '{"method": "mtp", "model": "google/gemma-4-12B-it-assistant",'
        ' "num_speculative_tokens": 1}' in block
    ), "vllm-multimodal --speculative-config must be the exact native-MTP config measured in §7"


def test_fleet_compose_multimodal_coder_is_opt_in_with_no_spec_decode() -> None:
    """vllm-multimodal-coder: opt-in (profiles: key), NO --speculative-config.

    The coder fine-tune is kept but demoted (§7): native MTP only reaches 30.8%
    draft acceptance on it — not worth wiring, unlike the default base gear above.
    """
    text = _fleet_compose_text()
    assert (
        "vllm-multimodal-coder:" in text
    ), "vllm-multimodal-coder service must be defined in fleet compose (opt-in coder gear)"
    block = _service_block(text, "vllm-multimodal-coder")
    assert (
        "profiles:" in block
    ), "vllm-multimodal-coder must be behind a profiles: key — it is opt-in"
    assert (
        "--speculative-config" not in block
    ), "vllm-multimodal-coder must NOT carry --speculative-config (only 30.8% draft accept; §6/§7)"
    assert _GEMMA_CODER_ID in block, f"vllm-multimodal-coder must reference {_GEMMA_CODER_ID!r}"


def test_fleet_compose_multimodal_forces_triton_attention() -> None:
    """vllm-multimodal sets VLLM_ATTENTION_BACKEND=TRITON_ATTN — Gemma4's non-square
    attention (global_head_dim 512 ≠ head_dim 256) needs it (#71)."""
    block = _service_block(_fleet_compose_text(), "vllm-multimodal")
    assert (
        "VLLM_ATTENTION_BACKEND" in block and "TRITON_ATTN" in block
    ), "vllm-multimodal must set VLLM_ATTENTION_BACKEND=TRITON_ATTN for Gemma4 non-square attention"


def test_fleet_compose_middle_is_behind_profile() -> None:
    """vllm-middle (14B, legacy candidate) is opt-in: must declare a profiles: key."""
    block = _service_block(_fleet_compose_text(), "vllm-middle")
    assert (
        "profiles:" in block
    ), "vllm-middle must be behind a profiles: key — it is a legacy opt-in candidate"


# --- t4: default-on generate/embed/rerank gears unify on the pinned nightly --
# digest (devague plan lobes-unifies-its-generate-lane-on-one-vllm-nightl,
# task t4). Same digest Dockerfile.vllm-gemma4 already bases off — see
# docs/vllm-nightly-migration.md §4/§5 for the t2/t3 spikes that validated it.

_PRE_T4_NGC_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"
_NIGHTLY_DIGEST_IMAGE = (
    "vllm/vllm-openai@sha256:" "7c5a10e9a8b3c8642f4d0463a41215176c0dd834b4f0967287c7e3e517cf1be9"
)


def _audio_compose_text() -> str:
    from importlib.resources import files

    return (files("lobes.templates") / "fleet" / "docker-compose.audio.yml").read_text(
        encoding="utf-8"
    )


def test_fleet_generate_embed_rerank_on_nightly() -> None:
    """t4: vllm-primary/vllm-embed/vllm-rerank now pin the SAME nightly digest
    Dockerfile.vllm-gemma4 already bases off — one engine, fleet-wide, for the
    default-on gears. vllm-multimodal was already nightly (via its Dockerfile
    build); the realtime sidecars + gateway are untouched (no vLLM at all)."""
    text = _fleet_compose_text()

    for service in ("vllm-primary", "vllm-embed", "vllm-rerank"):
        block = _service_block(text, service)
        assert (
            _NIGHTLY_DIGEST_IMAGE in block
        ), f"{service} must pin the nightly digest {_NIGHTLY_DIGEST_IMAGE!r} (t4 flip)"
        assert (
            _PRE_T4_NGC_IMAGE not in block
        ), f"{service} must no longer pin the pre-t4 image {_PRE_T4_NGC_IMAGE!r}"
        assert "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0" in block, (
            f"{service} must disable the nightly cudagraph memory estimate "
            "(the same gotcha vllm-multimodal already works around)"
        )

    # vllm-multimodal: already nightly, via its Dockerfile build (unchanged by t4).
    multimodal_block = _service_block(text, "vllm-multimodal")
    assert "Dockerfile.vllm-gemma4" in multimodal_block
    assert _PRE_T4_NGC_IMAGE not in multimodal_block

    # gateway: not a vLLM service at all — builds Dockerfile.gateway, untouched.
    gateway_block = _service_block(text, "gateway")
    assert _NIGHTLY_DIGEST_IMAGE not in gateway_block
    assert _PRE_T4_NGC_IMAGE not in gateway_block
    assert "Dockerfile.gateway" in gateway_block

    # Opt-in minor/middle gears are OUT OF SCOPE for t4 (that's task t8) — they
    # must still pin the pre-t4 NGC image, unchanged.
    for service in ("vllm-minor", "vllm-middle"):
        block = _service_block(text, service)
        assert _PRE_T4_NGC_IMAGE in block, (
            f"{service} is out of scope for t4 (trailing task t8) — must still pin "
            f"{_PRE_T4_NGC_IMAGE!r}"
        )
        assert _NIGHTLY_DIGEST_IMAGE not in block

    # Realtime sidecars (Parakeet/Chatterbox/realtime bridge) live in the audio
    # overlay compose — out of scope for t4, not vLLM at all, must be unchanged.
    audio_text = _audio_compose_text()
    for service in ("chatterbox", "stt", "realtime"):
        block = _service_block(audio_text, service)
        assert _NIGHTLY_DIGEST_IMAGE not in block
        assert "vllm" not in block.lower()
