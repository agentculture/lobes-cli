"""The packaged stt-he lane sidecar (stt-lanes): overlay, image, server, init.

Template-only — nothing here is deployed. The overlay declares ONE CPU-only,
profile-gated, port-less service; the image is python:3.12-slim + ffmpeg with
``uv pip install --system`` and pinned weights; the server's pure helpers
import offline; ``lobes init --stt-he`` ships the files and the compose chain
picks the overlay up whenever it is present.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

from lobes.cli import main
from lobes.runtime import _compose
from lobes.stt_lanes import lane_by_name

_FLEET = Path(__file__).resolve().parents[1] / "lobes" / "templates" / "fleet"
_SERVER = _FLEET / "listen_server_zipformer.py"
HE = lane_by_name("stt-he")


def _text(name: str) -> str:
    return (files("lobes.templates") / "fleet" / name).read_text(encoding="utf-8")


def _overlay() -> dict:
    return yaml.safe_load(_text("docker-compose.stt-he.yml"))["services"]


def _env(svc: dict) -> dict[str, str]:
    return dict(e.split("=", 1) for e in svc["environment"])


# --- the compose overlay ----------------------------------------------------------


def test_overlay_declares_exactly_the_stt_he_service():
    assert set(_overlay()) == {"stt-he"}


def test_service_is_cpu_only_profile_gated_and_port_less():
    svc = _overlay()["stt-he"]
    assert svc["container_name"] == "model-gear-stt-he"
    assert svc["profiles"] == ["stt-he"]
    assert svc["build"]["dockerfile"] == "Dockerfile.stt-zipformer"
    assert "deploy" not in svc and "runtime" not in svc  # no GPU reservation
    assert "ports" not in svc
    assert svc["expose"] == ["9002"]
    assert "depends_on" not in svc  # issue #222
    assert svc["mem_limit"] == "${STT_HE_MEM_LIMIT:-1.5g}"
    assert svc["cpus"] == "${STT_HE_CPUS:-2}"
    env = _env(svc)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_DISABLE_XET"] == "1"
    assert env["STT_HE_REVISION"] == f"${{STT_HE_REVISION:-{HE.revision}}}"


def test_healthcheck_and_gateway_probe_the_same_ready_path():
    from lobes.stt_lanes import READY_PATH

    test = " ".join(_overlay()["stt-he"]["healthcheck"]["test"])
    assert f"http://localhost:9002{READY_PATH}" in test


def test_overlay_is_never_run_with_an_env_file():
    # Explicit environment only: no fleet secrets reach a CPU sidecar.
    assert "env_file" not in _overlay()["stt-he"]


def test_base_gateway_passes_the_lane_keys_through():
    gateway = yaml.safe_load(_text("docker-compose.yml"))["services"]["gateway"]
    env = _env(gateway)
    for key in ("STT_HE_BASE_URL", "STT_HE_FEASIBLE", "STT_EN_BASE_URL", "STT_EN_FEASIBLE"):
        assert env[key] == f"${{{key}:-}}"  # empty default: inert until set


def test_env_example_documents_every_lane_knob():
    text = _text("env.example")
    for key in (
        "STT_HE_BASE_URL",
        "STT_HE_FEASIBLE",
        "STT_HE_MEM_LIMIT",
        "STT_HE_CPUS",
        "STT_EN_BASE_URL",
        "STT_EN_FEASIBLE",
    ):
        assert re.search(rf"^# {key}=", text, re.MULTILINE), key


# --- the image -------------------------------------------------------------------


def _instructions() -> str:
    return "\n".join(
        ln for ln in _text("Dockerfile.stt-zipformer").splitlines() if not ln.startswith("#")
    )


def test_image_is_slim_python_with_ffmpeg_and_no_cuda():
    text = _instructions()
    assert re.search(r"^FROM python:3\.12-slim$", text, re.MULTILINE)
    assert "ffmpeg" in text
    assert "cuda" not in text.lower().replace("cuda_visible_devices", "")


def test_image_installs_with_uv_never_plain_pip():
    text = _instructions()
    assert "uv pip install --system" in text
    assert not re.search(r"(^|&&\s*|RUN\s+)pip install", text, re.MULTILINE)
    assert '"sherpa-onnx==1.13.8"' in text


def test_image_copies_the_server_and_its_readiness_twin():
    text = _instructions()
    assert "COPY listen_server_zipformer.py /app/listen_server_zipformer.py" in text
    assert "COPY _readiness.py /app/_readiness.py" in text
    assert 'CMD ["python3", "/app/listen_server_zipformer.py"]' in text
    assert _overlay()["stt-he"]["command"] == ["python3", "/app/listen_server_zipformer.py"]
    assert "HF_HUB_DISABLE_XET=1" in text


# --- the server's pure helpers -----------------------------------------------------


@pytest.fixture(scope="module")
def lsz():
    spec = importlib.util.spec_from_file_location(_SERVER.stem, _SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def test_server_pins_the_registry_identity(lsz):
    assert lsz.MODEL_ID == HE.catalog_id
    assert lsz.MODEL_REVISION == HE.revision
    assert lsz.RUNTIME == HE.runtime
    assert lsz.LANGUAGE == HE.language


def test_model_paths_are_the_int8_sherpa_export(lsz):
    paths = lsz.model_paths("/m")
    assert paths == {
        "tokens": "/m/tokens.txt",
        "encoder": "/m/encoder.int8.onnx",
        "decoder": "/m/decoder.int8.onnx",
        "joiner": "/m/joiner.int8.onnx",
    }
    assert lsz.missing_model_files("/nonexistent") == list(lsz.model_paths("/nonexistent").values())


def test_ffmpeg_transcodes_to_16k_mono_pcm16(lsz):
    cmd = lsz.ffmpeg_command("/tmp/in.m4a")  # nosec B108 - never opened
    assert cmd[0] == "ffmpeg"
    assert cmd[cmd.index("-ar") + 1] == "16000"
    assert cmd[cmd.index("-ac") + 1] == "1"
    assert cmd[cmd.index("-f") + 1] == "s16le"


def test_split_even_keeps_the_odd_byte(lsz):
    assert lsz.split_even(b"abcde") == (b"abcd", b"e")
    assert lsz.split_even(b"ab") == (b"ab", b"")


@pytest.mark.parametrize("fmt", ["json", "text", "verbose_json"])
def test_supported_formats(lsz, fmt):
    assert lsz.response_format_error(fmt) is None


@pytest.mark.parametrize("fmt", ["srt", "vtt", "", None])
def test_unsupported_formats(lsz, fmt):
    assert "unsupported response_format" in lsz.response_format_error(fmt)


def test_response_shapes(lsz):
    assert lsz.shape_response("שלום", "json", 1.0) == ("json", {"text": "שלום"})
    assert lsz.shape_response("שלום", "text", 1.0) == ("text", "שלום")
    kind, body = lsz.shape_response("שלום", "verbose_json", 2.5)
    assert kind == "json"
    assert body == {"text": "שלום", "language": "he", "duration": 2.5, "model": HE.catalog_id}


def test_readiness_needs_the_model_and_a_warm_up(lsz):
    assert lsz.readiness(False, False)[0] == 503
    status, body = lsz.readiness(True, False)
    assert status == 503 and body["reason"] == "warm-up decode pending"
    status, body = lsz.readiness(True, True)
    assert status == 200
    assert body["status"] == "ready"
    assert body["model"] == HE.catalog_id and body["revision"] == HE.revision


# --- init + the compose chain ----------------------------------------------------------


def test_init_stt_he_ships_every_file_the_image_needs(tmp_path):
    target = tmp_path / "d"
    assert main(["init", str(target), "--stt-he", "--apply"]) == 0
    for name in set(_compose.STT_HE_TEMPLATES.values()):
        assert (target / name).is_file(), name
    text = (target / _compose.STT_HE_OVERLAY).read_text(encoding="utf-8")
    for ref in re.findall(r"^\s*dockerfile:\s*(\S+)", text, re.MULTILINE):
        assert (target / ref).is_file()


def test_plain_init_does_not_ship_the_overlay(tmp_path):
    target = tmp_path / "d"
    assert main(["init", str(target), "--apply"]) == 0
    assert not (target / _compose.STT_HE_OVERLAY).exists()


def test_init_stt_he_dry_run_lists_the_files(tmp_path, capsys):
    assert main(["init", str(tmp_path / "d"), "--stt-he", "--json"]) == 0
    names = {f["name"] for f in json.loads(capsys.readouterr().out)["files"]}
    assert set(_compose.STT_HE_TEMPLATES.values()) <= names


def test_init_stt_he_refused_with_single(capsys):
    assert main(["init", "/tmp/nope", "--single", "--stt-he"]) == 1  # nosec B108
    assert "--stt-he" in capsys.readouterr().err


def test_compose_chain_includes_the_overlay_when_present(tmp_path):
    assert _compose._compose_files(tmp_path) == []
    (tmp_path / _compose.STT_HE_OVERLAY).write_text("services: {}\n")
    assert _compose._compose_files(tmp_path) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.STT_HE_OVERLAY,
    ]
    assert _compose.compose_file_args(
        audio=True, shape=True, local=True, embed=True, stt_he=True
    ) == [
        "-f",
        _compose.COMPOSE_FILE,
        "-f",
        _compose.AUDIO_OVERLAY,
        "-f",
        _compose.EMBED_OVERLAY,
        "-f",
        _compose.STT_HE_OVERLAY,
        "-f",
        _compose.SHAPE_OVERLAY,
        "-f",
        _compose.LOCAL_OVERRIDE,
    ]
