"""Tests for the custom vLLM image wiring for the vllm-multimodal service (issue #71).

Asserts that:
  - vllm-multimodal gains a build: block (context: ., dockerfile: Dockerfile.vllm-gemma4)
    AND an image: override (${MULTIMODAL_IMAGE:-lobes/vllm-gemma4:local}).
  - vllm-primary, vllm-embed, and vllm-rerank keep image: nvcr.io/nvidia/vllm:26.04-py3
    unchanged.
  - docker-compose.audio.yml does NOT gain a build/image override for vllm-multimodal
    (the audio overlay is untouched by t2).
"""

from __future__ import annotations

from pathlib import Path

import yaml

_TEMPLATES = Path(__file__).resolve().parents[1] / "lobes" / "templates"
_FLEET_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.yml"
_AUDIO_COMPOSE = _TEMPLATES / "fleet" / "docker-compose.audio.yml"

_STOCK_VLLM_IMAGE = "nvcr.io/nvidia/vllm:26.04-py3"
_CUSTOM_DOCKERFILE = "Dockerfile.vllm-gemma4"
_LOCAL_TAG = "lobes/vllm-gemma4:local"
_MULTIMODAL_IMAGE_VAR = "MULTIMODAL_IMAGE"


def _load_fleet() -> dict:
    return yaml.safe_load(_FLEET_COMPOSE.read_text(encoding="utf-8"))


class TestMultimodalServiceHasBuildBlock:
    """vllm-multimodal must declare both build: and image: so operators can
    either build locally (lobes fleet up --build) or pull a registry tag
    (docker compose pull vllm-multimodal with MULTIMODAL_IMAGE set)."""

    def test_vllm_multimodal_has_build_key(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-multimodal"]
        assert "build" in svc, "vllm-multimodal must have a build: block"

    def test_vllm_multimodal_build_context_is_dot(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-multimodal"]
        build = svc["build"]
        assert (
            build.get("context") == "."
        ), f"build.context must be '.' (got {build.get('context')!r})"

    def test_vllm_multimodal_build_dockerfile_is_gemma4(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-multimodal"]
        build = svc["build"]
        assert build.get("dockerfile") == _CUSTOM_DOCKERFILE, (
            f"build.dockerfile must be {_CUSTOM_DOCKERFILE!r} " f"(got {build.get('dockerfile')!r})"
        )

    def test_vllm_multimodal_image_uses_multimodal_image_var(self) -> None:
        # The raw YAML text must contain the ${MULTIMODAL_IMAGE:-...} form because
        # PyYAML expands nothing — the variable placeholder must be visible in text.
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        assert (
            _MULTIMODAL_IMAGE_VAR in text
        ), f"${_MULTIMODAL_IMAGE_VAR} variable not found in fleet compose"

    def test_vllm_multimodal_image_local_fallback_tag(self) -> None:
        # The local fallback tag must appear so plain `docker compose up` uses it.
        text = _FLEET_COMPOSE.read_text(encoding="utf-8")
        assert _LOCAL_TAG in text, f"Local fallback tag {_LOCAL_TAG!r} not found in fleet compose"

    def test_vllm_multimodal_has_image_key_parsed(self) -> None:
        # PyYAML sees the raw string (with ${}), so image: value must start with
        # the MULTIMODAL_IMAGE variable reference pattern.
        compose = _load_fleet()
        svc = compose["services"]["vllm-multimodal"]
        assert "image" in svc, "vllm-multimodal must have an image: key"
        image_val: str = svc["image"]
        assert image_val.startswith(
            "${MULTIMODAL_IMAGE"
        ), f"vllm-multimodal image: must start with ${{MULTIMODAL_IMAGE (got {image_val!r})"


class TestOtherServicesUnchanged:
    """vllm-primary, vllm-embed, and vllm-rerank must keep the stock NGC image."""

    def test_vllm_primary_keeps_stock_image(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-primary"]
        assert svc.get("image") == _STOCK_VLLM_IMAGE, (
            f"vllm-primary must keep image: {_STOCK_VLLM_IMAGE!r} " f"(got {svc.get('image')!r})"
        )

    def test_vllm_embed_keeps_stock_image(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-embed"]
        assert svc.get("image") == _STOCK_VLLM_IMAGE, (
            f"vllm-embed must keep image: {_STOCK_VLLM_IMAGE!r} " f"(got {svc.get('image')!r})"
        )

    def test_vllm_rerank_keeps_stock_image(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-rerank"]
        assert svc.get("image") == _STOCK_VLLM_IMAGE, (
            f"vllm-rerank must keep image: {_STOCK_VLLM_IMAGE!r} " f"(got {svc.get('image')!r})"
        )

    def test_vllm_primary_has_no_build_key(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-primary"]
        assert "build" not in svc, "vllm-primary must NOT have a build: block"

    def test_vllm_embed_has_no_build_key(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-embed"]
        assert "build" not in svc, "vllm-embed must NOT have a build: block"

    def test_vllm_rerank_has_no_build_key(self) -> None:
        compose = _load_fleet()
        svc = compose["services"]["vllm-rerank"]
        assert "build" not in svc, "vllm-rerank must NOT have a build: block"


class TestAudioOverlayUntouched:
    """The audio overlay compose must not reference vllm-multimodal at all
    (it has no reason to override that service's image)."""

    def test_audio_compose_has_no_vllm_multimodal_service(self) -> None:
        audio = yaml.safe_load(_AUDIO_COMPOSE.read_text(encoding="utf-8"))
        services = audio.get("services", {})
        assert (
            "vllm-multimodal" not in services
        ), "docker-compose.audio.yml must not define a vllm-multimodal override"
