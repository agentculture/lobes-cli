"""Static assertions for lobes/templates/fleet/Dockerfile.comfyui.

No docker daemon, no network, no image builds — pure file-content checks, the
same contract as tests/test_gemma4_dockerfile.py.

The live build + FLUX render that these pins were taken from is recorded in
docs/evidence/2026-09-16-t2-comfyui-container-spark.txt (DGX Spark GB10,
2026-09-16). CI is CPU-only x86 and does NOT build this image, so these checks
are what keep the recipe from drifting away from the validated one between live
runs.
"""

import re
from pathlib import Path

DOCKERFILE = Path(__file__).parent.parent / "lobes" / "templates" / "fleet" / "Dockerfile.comfyui"


def _text() -> str:
    return DOCKERFILE.read_text(encoding="utf-8")


def test_dockerfile_exists() -> None:
    assert DOCKERFILE.exists(), f"Expected {DOCKERFILE} to exist"


def test_built_from_a_lean_cuda_base_not_a_pulled_comfyui_image() -> None:
    """No official ComfyUI image matches the validated venv, so the image is
    BUILT from the same lean CUDA runtime base Dockerfile.chatterbox uses —
    one that ships no torch, leaving the playbook's cu130 wheels the only ones
    present."""
    from_lines = [ln.strip() for ln in _text().splitlines() if ln.strip().startswith("FROM")]
    assert len(from_lines) == 1, f"Expected exactly one FROM, got: {from_lines}"
    assert from_lines[0].startswith(
        "FROM nvidia/cuda:"
    ), f"Expected a lean nvidia/cuda base, got: {from_lines[0]}"


def test_comfyui_pinned_to_v0_33_2_and_its_commit() -> None:
    """Acceptance criterion 1: the image pins ComfyUI v0.33.2. The commit
    assertion turns an upstream re-tag into a build failure rather than a
    silent code swap."""
    text = _text()
    assert "v0.33.2" in text, "ComfyUI is not pinned to v0.33.2"
    assert "7cee3ceb" in text, "the commit v0.33.2 must resolve to is not asserted"
    assert "git clone --branch" in text, "ComfyUI must be installed by clone-at-tag"


def test_torch_comes_from_the_stock_cu130_index() -> None:
    """Acceptance criterion 1: torch from the STOCK upstream cu130 index — not
    an NGC pytorch base, not a Jetson wheel index — matching the venv it
    replaces (torch 2.14.0+cu130 / torchvision 0.29.0+cu130)."""
    text = _text()
    assert (
        "--index-url https://download.pytorch.org/whl/cu130" in text
    ), "torch must be installed from the stock cu130 index"
    assert "torch==2.14.0+cu130" in text, "torch is not pinned to the venv's 2.14.0+cu130"
    assert (
        "torchvision==0.29.0+cu130" in text
    ), "torchvision is not pinned to the venv's 0.29.0+cu130"


def test_torch_is_installed_before_comfyui_requirements() -> None:
    """Ordering is load-bearing: ComfyUI's requirements.txt lists bare
    `torch`/`torchvision`, so installing it first would resolve CPU-only wheels
    from PyPI and the cu130 build would never land."""
    text = _text()
    torch_at = text.index("--index-url https://download.pytorch.org/whl/cu130")
    reqs_at = text.index("-r /opt/ComfyUI/requirements.txt")
    assert torch_at < reqs_at, "torch must be installed BEFORE ComfyUI's requirements.txt"


def test_no_third_party_custom_nodes_are_vendored() -> None:
    """The deployment this reproduces carries only ComfyUI's own
    websocket_image_save.py in custom_nodes, so stock ComfyUI is the whole of
    what needs reproducing — nothing may be copied or cloned into custom_nodes."""
    text = _text()
    assert not re.search(
        r"^\s*(COPY|ADD)\b", text, re.MULTILINE
    ), "no files should be COPY/ADDed into this image"
    assert "custom_nodes" not in "\n".join(
        ln for ln in text.splitlines() if not ln.lstrip().startswith("#")
    ), "no instruction may install into custom_nodes"


def test_runs_non_root_as_uid_1000() -> None:
    """The compose service runs this as user "1000:1000" (t4), so the checkout
    must already be owned by that uid or the first write fails."""
    text = _text()
    assert "USER 1000:1000" in text, "the image must run as a non-root uid"
    assert "chown -R 1000:1000 /opt/ComfyUI" in text, "the checkout must be owned by that uid"


def test_entrypoint_does_not_bind_loopback() -> None:
    """The bare venv's start-comfy.sh binds 127.0.0.1 because ComfyUI ships no
    authn. In a container that makes the lane unreachable from the gateway —
    the same property is answered by publishing no host port instead (t3)."""
    text = _text()
    instructions = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    assert "127.0.0.1" not in instructions, "the image must not bind loopback"
    assert '"--listen", "0.0.0.0"' in instructions, "the image must bind 0.0.0.0"
    assert "EXPOSE 8188" in instructions


def test_entrypoint_and_cmd_are_split_so_flags_stay_overridable() -> None:
    text = _text()
    assert 'ENTRYPOINT ["python3.12", "/opt/ComfyUI/main.py"]' in text
    assert re.search(
        r'^CMD \["--listen"', text, re.MULTILINE
    ), "flags belong in CMD, not ENTRYPOINT"


def test_no_model_weights_are_baked_in() -> None:
    """The ~65G weight tree is bind-mounted read-only at /opt/ComfyUI/models by
    the compose service; the image must not download any of it."""
    text = _text()
    instructions = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
    for forbidden in ("hf download", "huggingface-cli", "safetensors", "wget "):
        assert forbidden not in instructions, f"the image must not fetch weights ({forbidden})"
    assert "/opt/ComfyUI/models" in instructions, "the models mount point must exist in the image"


def test_build_stage_verification_is_a_single_logical_line() -> None:
    """A multi-line `python3 -c` WITHOUT backslash continuations makes Docker
    parse each body line as its own instruction — the same trap guarded for
    Dockerfile.vllm-gemma4."""
    lines = _text().splitlines()
    for i, line in enumerate(lines):
        if 'python3.12 -c "' in line:
            j = i
            while j < len(lines) and lines[j].rstrip().endswith("\\"):
                j += 1
            body = "\n".join(lines[i : j + 1])
            assert body.count('"') % 2 == 0, f"unterminated python3 -c body at line {i + 1}"
            break
    else:  # pragma: no cover - the RUN exists; this is the guard's own guard
        raise AssertionError("no build-stage python3 -c verification found")
