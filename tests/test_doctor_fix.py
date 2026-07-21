"""``lobes doctor`` scaffold-integrity / profile-staleness checks + the ``--fix``
heal lane (issue #119).

Both live incidents are reproduced as fixtures, per the spec's honesty
conditions (h14/h15):

* **2026-07-14 Thor** — a pre-#110 ``.env`` served for weeks missing the thor
  profile's SM_110 divergence knobs, ``/health`` green, rerank lane hanging.
* **2026-07-17 Spark** — a partial audio scaffold: ``docker-compose.audio.yml``
  present but ``Dockerfile.chatterbox`` / ``Dockerfile.realtime`` absent and no
  audio ``.env`` keys (``append_audio_env`` never ran), so the gateway had
  ``AUDIO_URL`` unset and TTS 404'd for hours. Neither ``init --audio --apply``
  (refuses) nor ``--force`` (clobbers ``.env``) could heal it.
"""

from __future__ import annotations

import json

from lobes.cli import main
from lobes.cli._commands.init import _apply_profile_env
from lobes.cli._runtime_ops import resolve_init_profile
from lobes.runtime import _compose, _detect, _env


def _card(resolved: str, name: str = "NVIDIA GB10", cc: str = "sm_121") -> object:
    return _detect.DetectedCard(
        resolved=resolved,
        device_name=name,
        compute_capability=cc,
        total_memory_gb=128.0,
        hostname="testbox",
        device_tree_model=None,
        sources={
            "device_name": "nvidia-smi",
            "compute_capability": "nvidia-smi",
            "total_memory_gb": "/proc/meminfo",
            "hostname": "socket.gethostname",
            "device_tree_model": "unavailable",
        },
    )


def _scaffold_fleet(path, *, audio: bool = False, profile: str = "spark"):
    """A complete fleet deployment, as ``lobes init --apply`` leaves it."""
    templates = dict(_compose.FLEET_TEMPLATES)
    if audio:
        templates.update(_compose.AUDIO_TEMPLATES)
    _compose.write_scaffold(path, force=True, templates=templates)
    _compose.write_plugin_file(path, force=True)
    if audio:
        _compose.append_audio_env(path)
    _env.set_env(path / ".env", "LOBES_PROFILE", profile)
    return path


def _drop_env_keys(env_path, *keys: str) -> None:
    """Remove whole ``KEY=`` lines — simulating a ``.env`` that predates them."""
    lines = env_path.read_text(encoding="utf-8").splitlines()
    kept = [ln for ln in lines if not any(ln.startswith(k + "=") for k in keys)]
    env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def _doctor_json(capsys, *args: str) -> dict:
    main(["doctor", "--json", *args])
    return json.loads(capsys.readouterr().out)


def _tree_bytes(root) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


# --- the two incident fixtures, caught -------------------------------------


def test_thor_pre_110_env_flags_missing_divergence_knobs(tmp_path, monkeypatch, capsys):
    """The 2026-07-14 Thor incident: the profile-required keys are ABSENT."""
    _scaffold_fleet(tmp_path, profile="thor")
    _drop_env_keys(
        tmp_path / ".env",
        "RERANK_ATTENTION_BACKEND",
        "RERANK_ENFORCE_EAGER",
        "EMBED_ATTENTION_BACKEND",
    )
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("thor", "NVIDIA Thor", "sm_110"))

    payload = _doctor_json(capsys)
    ids = {c["id"]: c for c in payload["checks"]}
    stale = ids["profile_staleness"]
    assert stale["passed"] is False
    assert stale["severity"] == "warn"
    assert "RERANK_ATTENTION_BACKEND" in stale["message"]
    assert "doctor --fix" in stale["remediation"]
    assert "--force" not in stale["remediation"]  # never the destructive path
    assert "RERANK_ATTENTION_BACKEND" in payload["fix_plan"]["env"]
    # warn severity: the run does not flip unhealthy on staleness alone.
    assert payload["healthy"] is True


def test_thor_template_default_where_profile_requires_divergence(tmp_path, monkeypatch, capsys):
    """The other stale mode: the key EXISTS but still carries the template
    default the profile is supposed to diverge from — reported, but never
    auto-fixed (rewriting an existing line is not doctor's to do)."""
    _scaffold_fleet(tmp_path, profile="thor")  # env.example ships spark defaults
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("thor", "NVIDIA Thor", "sm_110"))

    payload = _doctor_json(capsys)
    stale = {c["id"]: c for c in payload["checks"]}["profile_staleness"]
    assert stale["passed"] is False
    assert "template default" in stale["message"]
    assert "RERANK_ATTENTION_BACKEND" in stale["message"]
    assert "never" in stale["remediation"]  # ...rewrites an existing line
    # NOT in the fix plan: the key exists, and --fix is missing-only.
    assert "RERANK_ATTENTION_BACKEND" not in payload["fix_plan"]["env"]


def test_spark_partial_audio_scaffold_flags_files_and_keys(tmp_path, monkeypatch, capsys):
    """The 2026-07-17 Spark incident: audio overlay present, two Dockerfiles
    absent, zero audio .env keys (append_audio_env never ran)."""
    _scaffold_fleet(tmp_path, audio=True)
    (tmp_path / "Dockerfile.chatterbox").unlink()
    (tmp_path / "Dockerfile.realtime").unlink()
    _drop_env_keys(tmp_path / ".env", "AUDIO_URL", "CHATTERBOX_PORT", "PARAKEET_PORT")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("spark"))

    payload = _doctor_json(capsys)
    ids = {c["id"]: c for c in payload["checks"]}
    files = ids["scaffold_files"]
    assert files["passed"] is False
    assert "Dockerfile.chatterbox" in files["message"]
    assert "Dockerfile.realtime" in files["message"]
    assert "doctor --fix" in files["remediation"]
    assert set(payload["fix_plan"]["files"]) == {"Dockerfile.chatterbox", "Dockerfile.realtime"}
    assert "AUDIO_URL" in payload["fix_plan"]["env"]


# --- the heal lane -----------------------------------------------------------


def test_fix_without_apply_mutates_nothing(tmp_path, monkeypatch, capsys):
    _scaffold_fleet(tmp_path, audio=True)
    (tmp_path / "Dockerfile.chatterbox").unlink()
    _drop_env_keys(tmp_path / ".env", "AUDIO_URL")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("spark"))

    before = _tree_bytes(tmp_path)
    payload = _doctor_json(capsys, "--fix")
    assert payload["fix_plan"]["files"] == ["Dockerfile.chatterbox"]
    assert "AUDIO_URL" in payload["fix_plan"]["env"]
    assert _tree_bytes(tmp_path) == before  # read-only without --apply


def test_fix_apply_heals_missing_only_and_never_touches_existing_lines(
    tmp_path, monkeypatch, capsys
):
    _scaffold_fleet(tmp_path, audio=True)
    (tmp_path / "Dockerfile.chatterbox").unlink()
    (tmp_path / "Dockerfile.realtime").unlink()
    _drop_env_keys(tmp_path / ".env", "AUDIO_URL", "CHATTERBOX_PORT", "PARAKEET_PORT", "HF_CACHE")
    env_path = tmp_path / ".env"
    # The HF_CACHE gotcha: a key that IS present (operator-set) must survive —
    # appending a blank template default over it would win under compose's
    # env_file last-duplicate-wins semantics.
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write("HF_CACHE=/custom/hf-cache\n")
    env_before = env_path.read_text(encoding="utf-8")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("spark"))

    payload = _doctor_json(capsys, "--fix", "--apply")
    assert any(a == "wrote Dockerfile.chatterbox" for a in payload["fix_applied"])
    assert any(a == "appended AUDIO_URL" for a in payload["fix_applied"])
    assert not any("HF_CACHE" in a for a in payload["fix_applied"])

    # The report describes the AFTER state: both new checks now pass.
    ids = {c["id"]: c for c in payload["checks"]}
    assert ids["scaffold_files"]["passed"] is True
    assert ids["profile_staleness"]["passed"] is True

    # Missing files restored byte-equal to the packaged templates.
    from importlib.resources import files as resource_files

    root = resource_files("lobes.templates")
    for tname, dest in (
        ("fleet/Dockerfile.chatterbox", "Dockerfile.chatterbox"),
        ("fleet/Dockerfile.realtime", "Dockerfile.realtime"),
    ):
        assert (tmp_path / dest).read_text(encoding="utf-8") == _compose._read_template(root, tname)

    # Append-only .env heal: every pre-existing line survives verbatim, in place.
    env_after = env_path.read_text(encoding="utf-8")
    assert env_after.startswith(env_before)
    appended = env_after[len(env_before) :]
    assert "AUDIO_URL=" in appended
    assert "HF_CACHE" not in appended  # present key never re-appended
    assert env_after.count("HF_CACHE=") == 1

    # Idempotent: a second heal is a byte-identical no-op.
    snapshot = _tree_bytes(tmp_path)
    payload2 = _doctor_json(capsys, "--fix", "--apply")
    assert payload2["fix_applied"] == []
    assert _tree_bytes(tmp_path) == snapshot


def test_fix_apply_heals_missing_vad_keys_and_preserves_a_customised_one(
    tmp_path, monkeypatch, capsys
):
    """issue #149 (task t5): env.audio.example now documents VAD_THRESHOLD,
    VAD_SILENCE_MS, VAD_PREFIX_PADDING_MS, VAD_MAX_TURN_MS,
    DEFAULT_TURN_DETECTION, DEFAULT_AEC_MODE. A pre-PR2 scaffold predates all
    six — doctor --fix --apply must append the missing ones, and MUST NEVER
    touch an operator-customised VAD_THRESHOLD already sitting in .env (the
    append-only contract: a duplicate appended key would win under compose's
    env_file last-wins semantics and silently clobber the real value)."""
    _scaffold_fleet(tmp_path, audio=True)
    _drop_env_keys(
        tmp_path / ".env",
        "VAD_THRESHOLD",
        "VAD_SILENCE_MS",
        "VAD_PREFIX_PADDING_MS",
        "VAD_MAX_TURN_MS",
        "DEFAULT_TURN_DETECTION",
        "DEFAULT_AEC_MODE",
    )
    env_path = tmp_path / ".env"
    # Simulate an operator who already tuned VAD_THRESHOLD away from the
    # template default (0.5) before this heal ever existed — replacing the
    # scaffolded line in place (not a duplicate) is what a hand-edited .env
    # looks like.
    with env_path.open("a", encoding="utf-8") as fh:
        fh.write("VAD_THRESHOLD=0.7\n")
    env_before = env_path.read_text(encoding="utf-8")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("spark"))

    payload = _doctor_json(capsys, "--fix", "--apply")
    applied = payload["fix_applied"]
    for key in (
        "VAD_SILENCE_MS",
        "VAD_PREFIX_PADDING_MS",
        "VAD_MAX_TURN_MS",
        "DEFAULT_TURN_DETECTION",
        "DEFAULT_AEC_MODE",
    ):
        assert f"appended {key}" in applied, f"expected {key} to be healed, got {applied}"
    # The customised key was already present — never re-appended, never healed.
    assert not any("VAD_THRESHOLD" in a for a in applied)

    env_after = env_path.read_text(encoding="utf-8")
    assert env_after.startswith(env_before)  # every pre-existing line survives verbatim
    appended = env_after[len(env_before) :]
    assert "VAD_THRESHOLD" not in appended
    assert env_after.count("VAD_THRESHOLD=") == 1
    assert "VAD_THRESHOLD=0.7" in env_after  # the operator's value, untouched

    # Healed keys carry env.audio.example's documented defaults.
    assert "VAD_SILENCE_MS=600" in appended
    assert "VAD_PREFIX_PADDING_MS=300" in appended
    assert "VAD_MAX_TURN_MS=30000" in appended
    assert "DEFAULT_TURN_DETECTION=server_vad" in appended
    assert "DEFAULT_AEC_MODE=none" in appended


def test_apply_without_fix_is_a_user_error(tmp_path, monkeypatch, capsys):
    _scaffold_fleet(tmp_path)
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    rc = main(["doctor", "--apply"])
    assert rc != 0
    assert "--fix" in capsys.readouterr().err


# --- honesty of the diff classes ---------------------------------------------


def test_operator_override_downgrades_to_info(tmp_path, monkeypatch, capsys):
    """A value differing from BOTH the template default and the render is an
    operator override — the check stays green and says so."""
    _scaffold_fleet(tmp_path, profile="thor")
    profile, _card_, _warn = resolve_init_profile("thor", tmp_path)
    from lobes.cli._commands.init import DEFAULT_SHAPE
    from lobes.profiles.shape_render import render_shape
    from lobes.profiles.shapes import resolve_shape

    _apply_profile_env(
        tmp_path / ".env", dict(render_shape(resolve_shape(DEFAULT_SHAPE), profile).env)
    )
    _env.set_env(tmp_path / ".env", "PRIMARY_GPU_MEM_UTIL", "0.44")  # hand-tuned reclaim
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("thor", "NVIDIA Thor", "sm_110"))

    payload = _doctor_json(capsys)
    stale = {c["id"]: c for c in payload["checks"]}["profile_staleness"]
    assert stale["passed"] is True
    assert "operator-set" in stale["message"]


def test_shape_dropped_role_keys_are_not_demanded(tmp_path, monkeypatch, capsys):
    """A spark-lobe-style deployment (senses dropped) must not be flagged for
    lacking the dropped lobe's knobs — the shape's drop decision is honoured."""
    _scaffold_fleet(tmp_path)
    (tmp_path / _compose.SHAPE_OVERLAY).write_text(
        'services:\n  vllm-multimodal:\n    profiles: ["shape-dropped"]\n'
        "  gateway:\n    depends_on: !reset null\n",
        encoding="utf-8",
    )
    _drop_env_keys(tmp_path / ".env", "MULTIMODAL_GPU_MEM_UTIL", "MULTIMODAL_MAX_MODEL_LEN")
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)
    monkeypatch.setattr(_detect, "detect_card", lambda: _card("spark"))

    payload = _doctor_json(capsys)
    stale = {c["id"]: c for c in payload["checks"]}["profile_staleness"]
    assert stale["passed"] is True
    assert "MULTIMODAL" not in str(payload["fix_plan"]["env"])


def test_single_model_deployment_skips_the_fleet_checks(tmp_path, monkeypatch, capsys):
    _compose.write_scaffold(tmp_path, force=True)  # legacy single-model set
    monkeypatch.setenv("LOBES_DIR", str(tmp_path))
    monkeypatch.setattr(_compose, "docker_available", lambda: True)

    payload = _doctor_json(capsys)
    ids = {c["id"] for c in payload["checks"]}
    assert "scaffold_files" not in ids
    assert "profile_staleness" not in ids
    assert "fix_plan" not in payload
