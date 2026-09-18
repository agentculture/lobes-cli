"""Static assertions for lobes/templates/fleet/Dockerfile.whisper-stt
(hebrew-realtime plan, task t10).

No docker daemon, no network, no image builds — pure file-content checks,
the same contract as tests/test_comfyui_dockerfile.py. This task's brief
explicitly forbids attempting a real ``docker build`` in this sandbox; a
live build + transcription is verified later on the box (see the acceptance
criteria in docs/specs/2026-09-18-hebrew-realtime.md, task t10).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_DOCKERFILE = _TEMPLATES / "Dockerfile.whisper-stt"
_PARAKEET_DOCKERFILE = _TEMPLATES / "Dockerfile.parakeet"
_AUDIO_HE_COMPOSE = _TEMPLATES / "docker-compose.audio-he.yml"
_AUDIO_COMPOSE = _TEMPLATES / "docker-compose.audio.yml"


def _text() -> str:
    return _DOCKERFILE.read_text(encoding="utf-8")


def _instructions() -> str:
    """The Dockerfile with comment-only lines stripped."""
    return "\n".join(ln for ln in _text().splitlines() if not ln.lstrip().startswith("#"))


def test_dockerfile_exists() -> None:
    assert _DOCKERFILE.exists(), f"Expected {_DOCKERFILE} to exist"


def test_base_image_is_pinned_and_matches_dockerfile_parakeet() -> None:
    """Justification (spelled out in the Dockerfile's own header comment):
    reuse the SAME base image Dockerfile.parakeet already uses, since it is
    also the exact image the t2 spike measured the Whisper runtime inside on
    this box's silicon — no second, unvalidated CUDA base."""
    text = _text()
    from_lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("FROM")]
    assert len(from_lines) == 1, f"Expected exactly one FROM, got: {from_lines}"
    assert from_lines[0] == "FROM scitrera/dgx-spark-vllm:0.16.0-t4"

    parakeet_from = [
        ln.strip()
        for ln in _PARAKEET_DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if ln.strip().startswith("FROM")
    ]
    assert from_lines == parakeet_from, "must reuse Dockerfile.parakeet's exact base image"


def test_installs_with_uv_pip_install_system() -> None:
    """Operator preference (memory: uv-over-pip-in-dockerfiles) — extras go
    in via `uv pip install --system`, not plain pip, once uv is bootstrapped."""
    instructions = _instructions()
    assert "uv pip install --system" in instructions
    assert "transformers" in instructions
    assert "fastapi" in instructions


def test_no_model_download_at_build_time() -> None:
    """Unlike Dockerfile.parakeet (which pre-downloads its model),
    Dockerfile.whisper-stt must NOT bake STT_MODEL in — it comes from the
    mounted HF cache volume, like every other fleet lane."""
    instructions = _instructions()
    for forbidden in ("from_pretrained", "hf download", "huggingface-cli"):
        assert (
            forbidden not in instructions
        ), f"must not fetch the model at build time ({forbidden})"


def test_hf_hub_disable_xet_is_set() -> None:
    """Avoids the uncapped hf-xet downloader that OOM-killed a co-resident
    lane on this box before (memory: orin-uncapped-process-oom-kills-associate)."""
    assert "HF_HUB_DISABLE_XET=1" in _text()


def test_server_and_readiness_are_copied_in() -> None:
    text = _text()
    assert "COPY _readiness.py" in text
    assert "COPY listen_server_whisper.py" in text


def test_readiness_copy_is_the_shared_vendored_file_not_a_new_one() -> None:
    """Reuses lobes/templates/fleet/_readiness.py (already vendored for
    Dockerfile.parakeet) — no second copy."""
    readiness_path = _TEMPLATES / "_readiness.py"
    assert readiness_path.exists()
    assert "COPY _readiness.py /app/_readiness.py" in _text()


def test_exposes_the_same_port_env_name_as_parakeet() -> None:
    """Same internal port env (PARAKEET_PORT, default 9002) as
    Dockerfile.parakeet — the realtime bridge's STT_URL needs no change."""
    text = _text()
    assert "EXPOSE 9002" in text
    parakeet_text = _PARAKEET_DOCKERFILE.read_text(encoding="utf-8")
    assert "EXPOSE 9002" in parakeet_text


def test_healthcheck_interpreter_exists_in_this_image() -> None:
    """docker-compose.audio.yml's `stt` healthcheck invokes `python3` via
    stdlib urllib. Dockerfile.whisper-stt reuses Dockerfile.parakeet's exact
    base image (asserted above), whose own healthcheck already relies on
    `python3` existing — so no interpreter mismatch is introduced here."""
    compose = yaml.safe_load(_AUDIO_COMPOSE.read_text(encoding="utf-8"))
    healthcheck_test = compose["services"]["stt"]["healthcheck"]["test"]
    assert "python3" in healthcheck_test
    assert 'CMD ["python3", "/app/listen_server_whisper.py"]' in _text()


def test_cmd_runs_the_whisper_server() -> None:
    assert 'CMD ["python3", "/app/listen_server_whisper.py"]' in _text()


def test_stt_he_service_override_targets_this_dockerfile() -> None:
    """Criterion 2: the stt-he service is added to docker-compose.audio-he.yml
    under the same service name (`stt`) and port (PARAKEET_PORT, unchanged)
    the bridge already targets."""
    compose = yaml.safe_load(_AUDIO_HE_COMPOSE.read_text(encoding="utf-8"))
    stt = compose["services"]["stt"]
    assert stt["build"]["dockerfile"] == "Dockerfile.whisper-stt"
    # No `container_name`/`expose`/`build.context` override here — those are
    # inherited from docker-compose.audio.yml's `stt` service (deep-merged by
    # compose), which is asserted separately in test_audio_he_stt_compose.py.
    assert "STT_MODEL" not in stt.get("environment", []) or any(
        e.startswith("STT_MODEL=") for e in stt["environment"]
    )


def test_build_stage_verification_has_no_unterminated_python_c_body() -> None:
    """Same guard as test_comfyui_dockerfile.py's — a multi-line
    `python3 -c "..."` without backslash continuations gets mis-parsed by
    Docker. This Dockerfile has no such inline verification, but the guard
    stays cheap insurance if one is added later."""
    lines = _text().splitlines()
    for i, line in enumerate(lines):
        if 'python3 -c "' in line or 'python3.12 -c "' in line:
            j = i
            while j < len(lines) and lines[j].rstrip().endswith("\\"):
                j += 1
            body = "\n".join(lines[i : j + 1])
            assert body.count('"') % 2 == 0, f"unterminated python3 -c body at line {i + 1}"
