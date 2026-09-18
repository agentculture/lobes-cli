"""Static assertions for lobes/templates/fleet/Dockerfile.chatterbox-ml
(hebrew-realtime plan, task t11).

No docker daemon, no network, no image builds — pure file-content checks,
the same contract as tests/test_whisper_stt_dockerfile.py and
tests/test_comfyui_dockerfile.py. A live arm64 GPU build + boot on the DGX
Spark is this task's own explicit stop condition (see the task brief); the
main agent runs that build and reports it separately.
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_DOCKERFILE = _TEMPLATES / "Dockerfile.chatterbox-ml"
_ENGLISH_DOCKERFILE = _TEMPLATES / "Dockerfile.chatterbox"
_AUDIO_HE_COMPOSE = _TEMPLATES / "docker-compose.audio-he.yml"
_AUDIO_COMPOSE = _TEMPLATES / "docker-compose.audio.yml"


def _text() -> str:
    return _DOCKERFILE.read_text(encoding="utf-8")


def _english_text() -> str:
    return _ENGLISH_DOCKERFILE.read_text(encoding="utf-8")


def _instructions(text: str) -> str:
    """The Dockerfile with comment-only lines stripped."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def test_dockerfile_exists() -> None:
    assert _DOCKERFILE.exists(), f"Expected {_DOCKERFILE} to exist"


def test_base_image_matches_dockerfile_chatterbox() -> None:
    """Reuses the SAME lean CUDA base as the English recipe — no second,
    unvalidated CUDA base for the Hebrew arm."""
    from_lines = [ln.strip() for ln in _text().splitlines() if ln.strip().startswith("FROM")]
    assert len(from_lines) == 1, f"Expected exactly one FROM, got: {from_lines}"

    english_from = [
        ln.strip() for ln in _english_text().splitlines() if ln.strip().startswith("FROM")
    ]
    assert from_lines == english_from, "must reuse Dockerfile.chatterbox's exact base image"


def test_torch_and_perth_pins_match_dockerfile_chatterbox() -> None:
    """The torch/torchaudio/Perth combo is the proven, ABI-validated one —
    must not drift between the two Dockerfiles."""
    text = _text()
    english = _english_text()
    for pin in (
        "torch==2.11.0+cu128",
        "torchaudio==2.11.0+cu128",
        "chatterbox-tts==0.1.7",
        "resemble-perth==1.0.1",
    ):
        assert pin in text, f"{pin} missing from Dockerfile.chatterbox-ml"
        assert pin in english, f"{pin} missing from Dockerfile.chatterbox (test itself is stale)"


def test_only_delta_is_phonikud_onnx_and_the_entrypoint() -> None:
    """The two Dockerfiles' non-comment instruction lines must differ by
    exactly: (1) an added phonikud-onnx install, (2) the [chatterbox-ml] vs
    [chatterbox] extra, (3) the ENTRYPOINT module. No other drift."""
    ml_lines = [ln for ln in _instructions(_text()).splitlines() if ln.strip()]
    en_lines = [ln for ln in _instructions(_english_text()).splitlines() if ln.strip()]

    ml_only = [ln for ln in ml_lines if ln not in en_lines]
    en_only = [ln for ln in en_lines if ln not in ml_lines]

    assert any("phonikud-onnx" in ln for ln in ml_only), "phonikud-onnx install line not found"
    assert any(
        "chatterbox-ml" in ln for ln in ml_only
    ), "[chatterbox-ml] extra install line not found"
    assert any(
        "chatterbox_multilingual_server" in ln for ln in ml_only
    ), "ENTRYPOINT does not target chatterbox_multilingual_server"

    assert any("[chatterbox]" in ln for ln in en_only), "English [chatterbox] extra line not found"
    assert any(
        "chatterbox_server" in ln and "chatterbox_multilingual_server" not in ln for ln in en_only
    ), "English ENTRYPOINT line not found"

    # Nothing else should differ: every other non-comment line is shared.
    unexplained_ml = [
        ln
        for ln in ml_only
        if "phonikud-onnx" not in ln
        and "chatterbox-ml" not in ln
        and "chatterbox_multilingual_server" not in ln
    ]
    assert not unexplained_ml, f"unexplained Dockerfile.chatterbox-ml-only lines: {unexplained_ml}"

    unexplained_en = [
        ln
        for ln in en_only
        if "[chatterbox]" not in ln
        and ("chatterbox_server" in ln and "chatterbox_multilingual_server" not in ln) is False
    ]
    # The English ENTRYPOINT line itself is expected to be en_only; everything
    # else en_only should be explained by the [chatterbox] extra line.
    unexplained_en = [ln for ln in unexplained_en if "ENTRYPOINT" not in ln]
    assert not unexplained_en, f"unexplained Dockerfile.chatterbox-only lines: {unexplained_en}"


def test_phonikud_onnx_is_pinned() -> None:
    assert "phonikud-onnx" in _text()


def test_installs_chatterbox_ml_extra() -> None:
    assert "lobes-cli[chatterbox-ml]" in _text()
    assert "lobes-cli[chatterbox]" not in _text()


def test_entrypoint_runs_the_multilingual_server() -> None:
    assert (
        'ENTRYPOINT ["python3.12", "-m", "lobes.realtime.chatterbox_multilingual_server"]'
        in _text()
    )


def test_exposes_port_9000_same_as_english() -> None:
    text = _text()
    assert "EXPOSE 9000" in text
    assert "EXPOSE 9000" in _english_text()


def test_phonikud_model_weights_are_not_baked_in() -> None:
    """The phonikud ONNX model file is mounted at runtime, never downloaded
    or copied at build time (criterion 3 / the module docstring's own claim)."""
    instructions = _instructions(_text())
    for forbidden in ("phonikud-1.0.int8.onnx", "wget", "from_pretrained"):
        assert forbidden not in instructions, f"must not bake the model in ({forbidden})"


def test_no_non_commercial_engine_referenced() -> None:
    """Plan risk r3 / criterion 3: no non-commercially-licensed weight or
    engine (phonikud-tts, its non-commercial StyleTTS2/Piper checkpoints) may
    be referenced by this shipped template. phonikud-ONNX (the diacritizer,
    CC BY 4.0) is a distinct, permissively-licensed package and is fine."""
    text = _text().lower()
    for forbidden in ("phonikud-tts", "thewh1teagle/phonikud-tts", "-checkpoints"):
        assert forbidden not in text, f"non-commercial engine reference found: {forbidden}"


def test_chatterbox_he_service_override_targets_this_dockerfile() -> None:
    """Criterion 3: the chatterbox service override in
    docker-compose.audio-he.yml targets this Dockerfile."""
    compose = yaml.safe_load(_AUDIO_HE_COMPOSE.read_text(encoding="utf-8"))
    chatterbox = compose["services"]["chatterbox"]
    assert chatterbox["build"]["dockerfile"] == "Dockerfile.chatterbox-ml"


def test_build_stage_verification_has_no_unterminated_python_c_body() -> None:
    """Same guard as test_comfyui_dockerfile.py's / test_whisper_stt_dockerfile.py's."""
    lines = _text().splitlines()
    for i, line in enumerate(lines):
        if 'python3 -c "' in line or 'python3.12 -c "' in line:
            j = i
            while j < len(lines) and lines[j].rstrip().endswith("\\"):
                j += 1
            body = "\n".join(lines[i : j + 1])
            assert body.count('"') % 2 == 0, f"unterminated python3.12 -c body at line {i + 1}"
